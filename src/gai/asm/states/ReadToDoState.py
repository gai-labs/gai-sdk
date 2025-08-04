from gai.asm.base import StateBase
from gai.lib.logging import getLogger
import os
from pathlib import Path

logger = getLogger(__name__)


class ToDoFileNotFoundError(Exception):
    """
    Raised when the todo file is not found.
    """

    pass


class ReadToDoState(StateBase):
    """
    State for reading todo items from a markdown file.

    state schema:
    {
        "READ_TODO": {
            "module_path": "gai.asm.states",
            "class_name": "ReadToDoState",
            "title": "READ_TODO",
            "input_data": {
                "todo_path": {"type": "state_bag", "dependency": "todo_path", "optional": True},
            },
            "output_data": ["todo_content", "todo_file_path", "file_exists"],
        }
    }
    """

    def __init__(self, machine):
        super().__init__(machine)

        # Define output data
        self.todo_content = None
        self.todo_file_path = None
        self.file_exists = False

    async def _read_todo_from_file(self, file_path):
        """Read todo content from a markdown file."""
        try:
            file_path = Path(file_path)

            if not file_path.exists():
                logger.warning(f"Todo file not found: {file_path}")
                return None, False

            # Read the content from the file
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            logger.info(f"Successfully read todo content from {file_path}")
            return content, True

        except Exception as e:
            logger.error(f"Failed to read todo content from {file_path}: {e}")
            raise e

    async def run_async(self):
        # Get todo file path from state bag or use default
        todo_path = self.machine.state_bag.get("todo_path")
        if not todo_path:
            todo_path = os.path.expanduser("~/.gai/todo.md")

        # Store the file path being used
        self.todo_file_path = str(todo_path)

        try:
            # Read the todo content
            content, exists = await self._read_todo_from_file(todo_path)

            self.todo_content = content
            self.file_exists = exists

            # Store results in state bag
            self.machine.state_bag["todo_content"] = content
            self.machine.state_bag["todo_file_path"] = self.todo_file_path
            self.machine.state_bag["file_exists"] = exists

            if exists and content:
                logger.info(
                    f"Read {len(content)} characters from {self.todo_file_path}"
                )
            elif exists:
                logger.info(f"Todo file exists but is empty: {self.todo_file_path}")
            else:
                logger.info(f"Todo file does not exist: {self.todo_file_path}")

        except Exception as e:
            logger.error(f"Error reading todo file: {e}")
            # Set default values on error
            self.todo_content = None
            self.file_exists = False
            self.machine.state_bag["todo_content"] = None
            self.machine.state_bag["todo_file_path"] = self.todo_file_path
            self.machine.state_bag["file_exists"] = False
            raise e
