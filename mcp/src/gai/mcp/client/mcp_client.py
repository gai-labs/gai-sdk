import os
from typing import Union
from gai.lib.logging import getLogger
from gai.lib.config import GaiClientConfig
from mcp import ClientSession, StdioServerParameters
from mcp.types import CallToolResult
from mcp.client.stdio import stdio_client
from gai.lib.errors import ToolNameNotFoundException
logger = getLogger(__name__)

class McpClient:
    
    @classmethod
    def get_filepath(cls, url: str) -> str:
        """
        Extracts the file path from a given MCP File URI.
        
        Args:
            url (str): The MCP File URI to parse.
            
        Returns:
            str: The extracted file path.
            
        Raises:
            ValueError: If the URI scheme is not 'file'.
        """
        from urllib.parse import urlparse, unquote
        import os
        
        parsed = urlparse(url)
        if parsed.scheme != "file":
            raise ValueError(f"Invalid MCP Web URI scheme: {parsed.scheme}. Expected 'file'.")
        
        netloc = parsed.netloc
        path = parsed.path
        
        # Reject remote hosts
        if netloc and netloc not in ("", "localhost"):
            first_seg = netloc
            rest = path.lstrip("/")
            filepath = os.path.normpath(os.path.join(first_seg, rest))
        else:
            filepath = path
            
        # Decode %-escapes (e.g. %20 → space)
        filepath = unquote(filepath)    
        
        return filepath
        

    def __init__(self, client_config: GaiClientConfig):
        self.client_config=client_config
        
        # Get command from client_config
        
        self.command = client_config.extra["command"]
        
        # Get blacklisted tools from client_config
        
        self.blocked = client_config.extra.get("blocked", [])

        args = []
        if self.command == "node" or self.command == "python":
            # Derive absolute server path from config
            
            server_path = self.get_filepath(client_config.url)
            if not os.path.isabs(server_path):
                # Find path relative to this file
                server_path=os.path.abspath(os.path.join(os.path.dirname(__file__), server_path))
            if not os.path.exists(server_path):
                raise FileNotFoundError(f"Server path does not exist: {server_path}")
            self.server_path = server_path
            args = [self.server_path]

        extra_args = client_config.extra.get("args",None)
        if extra_args:
            args.extend(extra_args)
        self.args=args
        
        # if any of the args is a file path, expand it
        
        for i,arg in enumerate(self.args):
            if arg.startswith("$HOME"):
                self.args[i] = arg.replace("$HOME", os.environ["HOME"])
            elif arg.startswith("~/"):
                self.args[i] = os.path.expanduser(arg)
            
        logger.info(f"mcp_client.__init__: server command={self.command} args={self.args}")
        
        self.server_params = StdioServerParameters(
            command=self.command,
            args=self.args,
            env=None
        )

    async def list_tools(self) -> dict:
        try:
            async with stdio_client(self.server_params) as (read,write): # type: ignore
                async with ClientSession(read,write) as client:
                    
                    await client.initialize()
                    result = await client.list_tools()
                    if not result:
                        print("mcp_server.list_Tools: No result returned from the server.")
                        raise ValueError("No result returned from the server.")

                    # Convert from MCP tools list format to an OpenAI compatible format
                    
                    available_tools = [{
                        "type": "function",
                        "function": {
                            "name": tool.name,
                            "description": tool.description,
                            "parameters": tool.inputSchema
                        },
                    } for tool in result.tools if tool.name not in self.blocked]
                    
                    # Add a default 'text' tool if no tools are available or if all tools are blocked
                    
                    # available_tools.append({
                    #     "type": "function",
                    #     "function": {
                    #         "name": "text",
                    #         "description": """Choosing 'text' means you are not using any tool.""",
                    #     }   
                    # })
                    
                    tools_list = ""
                    for tool in available_tools:
                        tools_list += f'"{tool["function"]["name"]}": {tool["function"]["description"]}\n\n'
                    tools_dict = {tool["function"]["name"]: tool for tool in available_tools}                    
                    return tools_dict
        except Exception as e:
            print(f"mcp_server.list_tools: error={e}")
            raise e        

    async def call_tool(self, tool_name: str, **kwargs) -> CallToolResult:
        try:
            async with stdio_client(self.server_params) as (read,write):
                async with ClientSession(read,write) as client:
                    
                    await client.initialize()
                    result = await client.call_tool(tool_name,arguments=kwargs)
                    if not result:
                        print("mcp_server.scrape: No result returned from the server.")
                        raise ValueError("No result returned from the server.")
                    return result
        except Exception as e:
            print(f"mcp_server.search: error={e}")
            raise e        


class McpAggregatedClient:
    
    def __init__(self,mcp_clients:Union[list[McpClient],list[str]]):
        
        from gai.lib.utils import run_async_function
        
        # If a list of mcp server names are passed in instead of mcp clients, load the clients.
        if mcp_clients and isinstance(mcp_clients[0], str):
            from gai.lib.config import config_helper
            mcp_clients = [McpClient(client_config=config_helper.get_client_config(name)) 
                         for name in mcp_clients]
       
        # List tools from all MCP clients right upfront and cache them

        self.mcp_clients = mcp_clients
        self.tool_to_server_params = {}
        self.tools = {}        
        
        for mcp_client in self.mcp_clients:
            result = run_async_function(mcp_client.list_tools)
            for tool_name,tool in result.items():
                
                if tool_name != "text" and tool_name in self.tool_to_server_params:
                    logger.warning(f"McpAggregatedClient: Duplicate tool name {tool_name} found across MCP clients. Do not overwrite.")
                    continue
                
                self.tools[tool_name] = tool
                self.tool_to_server_params[tool_name] = mcp_client.server_params
        
        
    async def list_tools(self) -> list:
        """
        Lists all tools available across multiple MCP servers.
        Returns:
            list: A list of tools including descriptions and parameters.
        """
        tools = []
        for tool in self.tools.values():
            tools.append({
                "name": tool["function"]["name"],
                "description": tool["function"]["description"],
                "input_schema": tool["function"]["parameters"]
            })
        return tools    
        
    async def call_tool(self, tool_name: str, **kwargs) -> CallToolResult:
        
        # This is a wrapper around the MCP clients to call a tool by name that will work across multiple MCP clients.
        
        if tool_name not in self.tool_to_server_params:
            raise ToolNameNotFoundException(tool_name=tool_name)
        
        server_params = self.tool_to_server_params[tool_name]
        
        for mcp_client in self.mcp_clients:
            if mcp_client.server_params == server_params:
                return await mcp_client.call_tool(tool_name, **kwargs)
        
        raise ValueError(f"Server parameters for tool {tool_name} not found in any MCP client.")
                
            

        