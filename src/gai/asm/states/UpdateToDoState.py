from gai.asm.base import StateBase
from gai.lib.logging import getLogger
from gai.llm.lib import LLMGeneratorRetryPolicy
from gai.llm.openai import AsyncOpenAI
import os
from pathlib import Path

logger = getLogger(__name__)


class MissingToDoDataError(Exception):
    """
    Raised when todo data is missing from the state bag.
    """

    pass


class UpdateToDoState(StateBase):
    """
    State for updating todo items using LLM assistance.

    state schema:
    {
        "UPDATE_TODO": {
            "module_path": "gai.asm.states",
            "class_name": "UpdateToDoState",
            "title": "UPDATE_TODO",
            "input_data": {
                "llm_config": {"type": "state_bag", "dependency": "llm_config"},
                "todo_data": {"type": "state_bag", "dependency": "todo_data"},
                "update_instruction": {"type": "state_bag", "dependency": "update_instruction"},
                "todo_path": {"type": "state_bag", "dependency": "todo_path", "optional": True},
                "mcp_server_names": {
                    "type": "state_bag",
                    "dependency": "mcp_server_names",
                },
            },
            "output_data": ["streamer", "get_assistant_message", "updated_todo", "todo_path"],
        }
    }
    """

    def __init__(self, machine):
        super().__init__(machine)

        # Define output data
        self.updated_todo = None
        self.todo_file_path = None

    async def _write_todo_to_file(self, content, file_path):
        """Write the updated todo content to a markdown file."""
        try:
            # Ensure the directory exists
            file_path = Path(file_path)
            file_path.parent.mkdir(parents=True, exist_ok=True)

            # Write the content to the file
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)

            logger.info(f"Successfully wrote todo content to {file_path}")
            return str(file_path)
        except Exception as e:
            logger.error(f"Failed to write todo content to {file_path}: {e}")
            raise e

    async def _raw_llm_stream(self, llm_client, llm_model, messages, tools):
        """Call the LLM once and yield raw (extracted) chunks."""

        async def _gen():
            resp = await llm_client.chat.completions.create(
                model=llm_model,
                messages=messages,
                tools=tools,
                stream=True,
            )

            async for chunk in resp:
                if chunk.choices and chunk.choices[0].delta:
                    yield chunk.choices[0].delta

        return LLMGeneratorRetryPolicy(self.machine).run(_gen)

    def _make_streamer(self, llm_client, llm_model, messages, tools):
        """
        Main function is to stream text followed by the last chunk for the completed object.
        """

        from gai.messages import message_helper

        chat_messages = message_helper.convert_to_chat_messages(messages)
        chat_messages = message_helper.shrink_messages(chat_messages)

        async def _streamer():
            async def process_last_chunk(last_chunk):
                self.machine.state_bag["get_assistant_message"] = None
                self.machine.state_bag["user_message"] = None

                if not isinstance(last_chunk, list):
                    raise TypeError(
                        f"UpdateToDoState._streamer: Expected last chunk to be a list, got {type(last_chunk)} instead."
                    )

                if last_chunk:
                    # record assistant message, history & getter
                    self.machine.monologue.add_assistant_message(
                        state=self, content=last_chunk
                    )

                    self.machine.state_history[-1]["output"]["monologue"] = (
                        self.machine.monologue.copy()
                    )

                    for item in last_chunk:
                        if item.get("type") == "text":
                            # The final text contain the complete assistant message
                            text = item.get("text", "")
                            self.machine.state_bag["get_assistant_message"] = (
                                lambda: text
                            )

                            # Store the updated todo
                            self.updated_todo = {
                                "original_todo": self.machine.state_bag.get(
                                    "todo_data"
                                ),
                                "update_instruction": self.machine.state_bag.get(
                                    "update_instruction"
                                ),
                                "updated_content": text,
                            }

                            # Write the updated content to file
                            todo_path = self.machine.state_bag.get("todo_path")
                            if not todo_path:
                                # Default path
                                todo_path = os.path.expanduser("~/.gai/todo.md")

                            try:
                                self.todo_file_path = await self._write_todo_to_file(
                                    text, todo_path
                                )
                                logger.info(
                                    f"Todo content written to: {self.todo_file_path}"
                                )
                            except Exception as e:
                                logger.error(f"Failed to write todo file: {e}")
                                # Don't raise the exception, just log it so the state can continue

            # Start streaming
            logger.debug("UpdateToDoState: Starting LLM streaming")
            chunks = []

            # Get the raw stream generator
            raw_stream_gen = await self._raw_llm_stream(
                llm_client, llm_model, chat_messages, tools
            )

            async for chunk in raw_stream_gen:
                chunks.append(chunk)

                # Yield the chunk
                yield {"type": "chunk", "content": chunk}

            # Process the final accumulated chunks
            await process_last_chunk(chunks)

        return _streamer()

    def get_assistant_message(self):
        """Return the latest assistant message from the monologue."""
        return self.machine.monologue.get_last_assistant_message()

    async def run_async(self):
        # Validate required input data from state bag
        todo_data = self.machine.state_bag.get("todo_data")
        if not todo_data:
            raise MissingToDoDataError(
                "UpdateToDoState.run_async: todo_data is missing from state bag."
            )

        update_instruction = self.machine.state_bag.get("update_instruction")
        if not update_instruction:
            raise MissingToDoDataError(
                "UpdateToDoState.run_async: update_instruction is missing from state bag."
            )

        # Get llm client
        llm_config = self.machine.state_bag["llm_config"]
        llm_client = AsyncOpenAI(llm_config)

        # Get model
        llm_model = llm_config["model"]

        # Get todo file path
        todo_path = self.machine.state_bag.get("todo_path")
        if not todo_path:
            todo_path = os.path.expanduser("~/.gai/todo.md")

        # Create system message for todo update
        system_message = f"""
            You are a helpful assistant specialized in updating and managing todo items.
            
            Current Todo Data:
            {todo_data}
            
            Update Instruction:
            {update_instruction}
            
            Please update the todo item according to the instruction provided. 
            Maintain the structure and format of the original todo data while applying the requested changes.
            If the instruction is unclear, ask for clarification using the appropriate tools.
            
            Provide the updated todo item in your response. The updated content will be automatically 
            saved to: {todo_path}
            
            Format your response as clean markdown content that can be directly written to the file.
            """

        # Get MCP client and tools
        mcp_client = self.machine.state_bag.get("mcp_client")
        tools = []
        if mcp_client:
            tools = await mcp_client.list_tools()

        # Add todo update message to monologue
        self.machine.monologue.add_user_message(state=self, content=system_message)

        messages = self.machine.monologue.list_messages()

        # Set up the streamer for LLM response
        self.machine.state_bag["streamer"] = self._make_streamer(
            llm_client, llm_model, messages, tools
        )

        # Store the target todo file path in state bag for reference
        self.machine.state_bag["todo_file_path"] = todo_path
