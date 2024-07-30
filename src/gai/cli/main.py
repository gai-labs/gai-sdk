import cmd,os
from openai import OpenAI
from gai.ttt.client.completions import Completions
from gai.cli._gai_cli.Init import Init
from gai.cli._gai_cli.Tools import Tools
from gai.cli._gai_cli.MultilineInputCmd import MultilineInputCmd
from gai.ttt.client.completions import Completions
from dotenv import load_dotenv
load_dotenv()
os.environ["LOG_LEVEL"]="ERROR"

from rich.console import Console
console=Console()
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.live import Live

# Help
HELP = '''
1. chat
2. search
3. export
4. import
5. load <agent>
'''

import threading
from time import sleep
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
console = Console()

def spinner_task(stop_event):
    with Progress(
        SpinnerColumn(), TextColumn("Processing..."), console=console
    ) as progress:
        task = progress.add_task("Processing...", start=True)
        while not stop_event.is_set():
            sleep(0.1)  # Spinner updates itself
        progress.update(task, advance=1)  # Completing task stops the spinner



class Cli(cmd.Cmd,
          Init,
          MultilineInputCmd,
          Tools
    ):

    def __init__(self):
        if os.environ.get("OPENAI_API_KEY") is None:
            os.environ["OPENAI_API_KEY"] = ""
        self.client = OpenAI()
        self.client = Completions.PatchOpenAI(self.client)

        self.intro = f'Welcome to Gaiaio Command-Line Interpreter (ver. {self._get_version()}).\nYour current directory path is {os.path.abspath(os.path.curdir)}.\nType help or ? to list commands.\n'    
        self.prompt = 'gai> '
        self.messages = None

        MultilineInputCmd.__init__(self)
        Init.__init__(self)

    # Usage: exit'
    def do_exit(self, arg):
        return True

    # Usage: help'
    def do_help(self, arg):
        print(HELP)

    def emptyline(self):
        """Handle the case where an empty line is entered."""
        pass    

    def default(self, message):
                
        self.messages.append({"role": "user", "content": message})
        self.messages.append({"role": "assistant", "content": ""})
        with console.status("Working...",spinner="monkey") as status:
            #
            content = ""
            response = self.client.chat.completions.create(model="exllamav2-mistral7b", messages=self.messages, stream=True)
            for chunk in response:
                # Stop the spinner on the first iteration
                if 'status' in locals():
                    status.stop()
                    del status                
                chunk = chunk.extract()
                if chunk and isinstance(chunk, str):
                    
                    # Skip empty chunks
                    if content is None and chunk==" ":
                        continue

                    content += chunk
                    console.print(f"[italic bright_white]{chunk}[/]",end="")
                    if self.messages:
                        self.messages.pop()
                    self.messages.append({"role": "assistant", "content": content})
                    self.save()
        print()


    def do_messages(self,ignored):
        print(self.messages)

# run
if __name__ == '__main__':
    Cli().cmdloop()