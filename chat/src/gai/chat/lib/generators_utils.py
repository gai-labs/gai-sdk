# A simple utility to validate if all items in model params are in the whitelist.
from typing import List, Union

from gai.lib.config import config_helper, DownloadConfig, GaiGeneratorConfig
from gai.lib.logging import getLogger
logger = getLogger(__name__)

def validate_params(model_params,whitelist_params):
    for key in model_params:
        if key not in whitelist_params:
            raise Exception(f"Invalid param '{key}'. Valid params are: {whitelist_params}")

# A simple utility to filter items in model params that are also in the whitelist.
def filter_params(model_params,whitelist_params):
    filtered_params={}
    for key in model_params:
        if key in whitelist_params:
            filtered_params[key]=model_params[key]
    return filtered_params

def has_ai_placeholder(messages):
    message = messages[-1]
    if message["role"].lower() != "system" and message["role"].lower() != "user" and message["content"] == "":
        return True
    return False

def get_tools_schema():
    return {
        "type": "object",
        "properties": {
            "function": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string"
                    },
                    "arguments": {
                        "type": "object",
                    }
                },
                "required": ["name", "arguments"]
            },
        },
        "required": ["function"],
        "additionalProperties": True
    }

def apply_tools_message( messages: List, tools:dict, tool_choice:str):
    # Proceed only if tools are available
    if not tools:
        return messages

    # Check if tools are required and add a tools prompt
    if tools:
        if tool_choice == "none":
            # If tool_choice == "none", skip adding tools
            return messages
        
        if tool_choice == "required":
            # Create a system message to introduce the tools
            system_message = {"role":"system","content":
            """
            1. Select the most probable tool from the list below that is most suitable for responding to the user's message and respond only in JSON and nothing else.

                {tools}

            2. You must return one tool from the list and must not respond with anything else.
            """}   

        # When tool_choice == "auto", the system can return a tool response
        # or a text response.
        if tool_choice == "auto":

            # Create a system message to introduce the tools
            system_message = {"role":"system","content":
            """
            1. Review the <tools> below and assess if any of them is suitable for responding to the user's message.

                {tools}

            2. If none of the tools are suitable, you can respond with a <text> response that looks like the following:
                
            {{
                "function": {{
                    "name": "text",
                    "arguments": {{
                        "text": "This is a text response."
                    }}
                }}
            }}
            """}

        # Create system_message
        system_message["content"] = system_message["content"].format(
            tools=tools)

        # Insert the system message immediately before the last user_message.                
        ai_placeholder = None
        if has_ai_placeholder(messages):
            ai_placeholder = messages.pop()
        user_message = messages.pop()
        messages.append(system_message)
        messages.append(user_message)
        if ai_placeholder:
            messages.append(ai_placeholder)

    return messages

def apply_schema_prompt( messages: List, schema):

    # Apply schema. Note that tool schema will override any provided schema.
    if schema:
        system_message={"role":"system","content":f"""Begin your response with an open curly brace. Your response must be parseable by this json schema: {schema} """}
        #system_message={"role":"system","content":f"You will respond to the user's message based only on the following JSON schema {schema}. Begin your response with a curly bracket '{{' and end it with a curly bracket '}}'."}

        # Insert the system message immediately before the last user_message.                
        ai_placeholder = None
        if has_ai_placeholder(messages):
            ai_placeholder = messages.pop()
        user_message = messages.pop()
        messages.append(system_message)
        messages.append(user_message)
        if ai_placeholder:
            messages.append(ai_placeholder)

    return messages

def format_list_to_prompt(messages, format_type="none",stream=False):
    prompt=""

    if format_type == "none":
        for message in messages:
            role = message['role']
            content = message['content']
            if content:
                prompt+=f"{role}: {content}\n"
            else:
                if role == "assistant":
                    prompt+=f"{role}: "
        return prompt

    if messages[-1]["content"]:
        raise Exception("Last message should be an AI placeholder")
    if format_type == "llama3":
        prompt="<|begin_of_text|>"
        for message in messages:
            role = message['role']
            role_prompt=f"<|start_header_id|>{role}<|end_header_id|>"
            content = message['content']
            if content:
                prompt+=f"{role_prompt}\n\n{content}<|eot_id|>"
            else:
                prompt+=role_prompt
        return prompt

    if format_type == "mistral":
        # I might be paranoid but somehow it seemed that different prompt format may be required for stream vs generation.

        if stream:
            prompt="<s>"
            for message in messages:
                role = message['role']
                content = message['content']
                if content:
                    prompt+=f"{role}: {content}\n"
                    if role.lower() == "assistant":
                        prompt+="</s><s>"
                else:
                    prompt+=f"{role}:"
            return prompt

        if not stream:
            # According to bartowski, the prompt format is: <s>[INST]  {prompt} [/INST]</s> and doesn't support system message.
            prompt="<s>"
            for message in messages:
                role = message['role']
                content = message['content']
                if role.lower() == "system" or role.lower() == "user":
                    if content:
                        prompt+=f"[INST]   {content}  [/INST]"
                if role.lower() == "assistant":
                    if content:
                        prompt+=f"{content}</s><s>"
            prompt.replace("[/INST][INST]","")
            return prompt

    raise Exception(f"Invalid format type '{format_type}'")

async def word_streamer_async( char_generator):
    buffer = ""
    async for byte_chunk in char_generator:
        if type(byte_chunk) == bytes:
            byte_chunk = byte_chunk.decode("utf-8", "replace")
        buffer += byte_chunk
        words = buffer.split(" ")
        if len(words) > 1:
            for word in words[:-1]:
                yield word
                yield " "
            buffer = words[-1]
    yield buffer            

def word_streamer( char_generator):
    buffer = ""
    for chunk in char_generator:
        if chunk:
            if type(chunk) == bytes:
                chunk = chunk.decode("utf-8", "replace")
            buffer += chunk
            words = buffer.split(" ")
            if len(words) > 1:
                for word in words[:-1]:
                    yield word
                    yield " "
                buffer = words[-1]
    yield buffer

    
    # if isinstance(name_or_config, str):
    #     generator_name = name_or_config
        
    #     # generator config from global config
    #     Exllamav2GeneratorConfig.get_generator_config(generator_name=generator_name)
    #     if not generator_config.source:
    #         raise Exception(f"Generator '{generator_name}' does not have a source defined.")
    #     name_or_config = generator_config.source

    # if isinstance(name_or_config, DownloadConfig):
    #     return name_or_config
        
    # if isinstance(name_or_config, dict):
    #     if "type" in name_or_config:
    #         if name_or_config["type"] == "huggingface":
    #             download_config = HuggingfaceDownloadConfig(**name_or_config)
    #         elif name_or_config["type"] == "civictai":
    #             download_config = CivitaiDownloadConfig(**name_or_config)
    #         else:
    #             raise Exception(f"Invalid generator config type '{name_or_config['type']}'. Must be 'huggingface' or 'civictai'.")
    #     return download_config
    # else:
    #     raise Exception(f"Invalid generator config type '{type(name_or_config)}'. Must be str, GaiGeneratorConfig or dict.")

def progress_bar_callback(status):
    """
    A callback function that prints download progress in a human-readable format
    
    Args:
        status: Dictionary containing progress information:
            - progress: percentage complete (0-100)
            - current: current file number
            - total: total number of files
            - filename: name of current file (if available)
            - message: status message
    """
    if not status:
        return
        
    # Format the progress bar
    if "progress" in status:
        progress = status["progress"]
        bar_length = 30
        filled_length = int(round(bar_length * progress / 100))
        
        bar = '■' * filled_length + '□' * (bar_length - filled_length)
        
        # Get additional information
        current = status.get("current", 0)
        total = status.get("total", 0)
        filename = status.get("filename", "")
        message = status.get("message", "")
        
        # Print the progress bar with file information
        import sys
        sys.stdout.write(f"\r{message}: [{bar}] {progress:.1f}% ({current}/{total}) {filename}")
        sys.stdout.flush()
        
        # Add newline when complete
        if progress >= 100:
            print()
    else:
        # Just print the message if no progress data
        print(status.get("message", str(status)))    

def text_progress_callback(status):
    """
    A minimal callback that directly prints status updates
    without any formatting or terminal manipulation.
    """
    if not status:
        return
    
    # For debug purposes, just print each update on its own line
    print(f"Download status: {status}")

def download(name_or_config: Union[str,GaiGeneratorConfig,dict], status_callback=None):
    from gai.lib.utils import get_app_path
    from huggingface_hub import snapshot_download
    import os
    
    app_dir = get_app_path()
    if name_or_config is GaiGeneratorConfig:
        model = name_or_config.source
    else:
        model = config_helper.get_download_config(name_or_config)
    
    if model.type=="huggingface":
        import shutil
        local_dir=f"{app_dir}/models/"+model.local_dir
        shutil.rmtree(local_dir, ignore_errors=True)
        
        # Force clean download to trigger callbacks
        download_kwargs = {
            "repo_id": model.repo_id,
            "local_dir": local_dir,
            "revision": model.revision,
            "force_download": True,  # Force redownload
            "resume_download": False  # Start fresh
        }
        
        if model.file:
            download_kwargs["allow_patterns"] = model.file
        
        # Simple, reliable tqdm override
        if status_callback:
            from tqdm.auto import tqdm
            
            class SimpleCallbackTqdm(tqdm):
                def update(self, n=1):
                    super().update(n)
                    if status_callback:
                        status_callback({
                            "progress": self.n/self.total*100 if self.total else 0,
                            "current": self.n, 
                            "total": self.total,
                            "message": "Downloading"
                        })
                    return True
            
            download_kwargs["tqdm_class"] = SimpleCallbackTqdm
        
        # Download the model
        snapshot_download(**download_kwargs)
        
    return local_dir