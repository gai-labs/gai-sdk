#!/bin/env python3
from pathlib import Path
import json
import sys,os
from openai import OpenAI
from rich.console import Console
import shutil

from gai.ttt.client.completions import Completions
console=Console()
from time import sleep
import subprocess

os.environ["HF_HUB_ENABLED_HF_TRANSFER"]="1"
os.environ["OPENAI_API_KEY"]=""
if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"]=""
client=OpenAI(api_key=os.environ["OPENAI_API_KEY"])
client=Completions.PatchOpenAI(client)

from gai.lib.common.utils import this_dir
here = this_dir(__file__)

def app_dir():
    with open(Path("~/.gairc").expanduser(), "r") as file:
        rc=file.read()
        jsoned = json.loads(rc)
        return Path(jsoned["app_dir"]).expanduser()

def init(force=False):

    if not force and (Path("~/.gairc").expanduser().exists() or Path("~/.gai").expanduser().exists()):
        console.print("[red]Already exists[/]")
        return
    
    # app_dir doesn't exist
    if not Path("~/.gai").expanduser().exists():
        Path("~/.gai").expanduser().mkdir()

    # models directory doesn't exist
    if not Path("~/.gai/models").expanduser().exists():
        Path("~/.gai/models").expanduser().mkdir()

    # Force create .gairc
    with open(Path("~/.gairc").expanduser(), "w") as f:
        f.write(json.dumps({
            "app_dir":"~/.gai"
        }))

    # Finally
    if Path("~/.gairc").expanduser().exists() and Path("~/.gai").expanduser().exists():
        console.print("[green]Initialized[/]")
    shutil.copy2(Path(__file__).parent.parent.parent.parent.parent / "gai.yml", Path("~/.gai").expanduser())
    

# requires pip install huggingface-hub
from huggingface_hub import hf_hub_download,snapshot_download
import time
def pull(model_name):
    
    if not model_name:
        console.print("[red]Model name not provided[/]")
        return
    if model_name == "exllamav2-mistral7b":
        start=time.time()
        console.print(f"[white]Downloading... {model_name}[/]")
        snapshot_download(
            repo_id="bartowski/Mistral-7B-Instruct-v0.3-exl2",
            local_dir=f"{app_dir()}/models/exllamav2-mistral7b",
            revision="1a09a351a5fb5a356102bfca2d26507cdab11111",
            )
        #os.removedirs(Path("~/.cache/huggingface").expanduser())
        end=time.time()
        duration=end-start
        model_name="exllamav2-mistral7b"
        download_size=Path(f"{app_dir()}/models/{model_name}").stat().st_size

        from rich.table import Table
        table = Table(title="Download Information")
        # Add columns
        table.add_column("Model Name", justify="left", style="bold yellow")
        table.add_column("Time Taken (s)", justify="right", style="bright_green")
        table.add_column("Size (Mb)", justify="right", style="bright_green")
        table.add_column("Location", justify="right", style="bright_green")

        # Add row with data
        table.add_row(model_name, f"{duration:4}", f"{download_size:2}", f"{app_dir()}/models/{model_name}")

        # Print the table to the console
        console.print(table)        

    else:
        console.print(f"[red]Model {model_name} not found[/]")

def docker_start():
    try:
        dest = os.path.join(here,"../../../../.vscode")
        docker_command = f"cd {dest} && docker-compose up -d --build --force-recreate"
        subprocess.run(docker_command, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error occurred: {e}")

def docker_stop():
    try:
        dest = os.path.join(here,"../../../../.vscode")
        docker_command = f"cd {dest} && docker-compose down -v --remove-orphans"
        result=subprocess.run(docker_command, shell=True, check=True, capture_output=True)
        # Print stdout and stderr for debugging
        print("STDOUT:", result.stdout.decode())
        print("STDERR:", result.stderr.decode())
        print("Containers and associated resources have been forcefully shut down and removed.")        
    except subprocess.CalledProcessError as e:
        print(f"Error occurred: {e.stderr.decode()}")

def news(news_url="https://asiaone.com"):
    from gai.cli._gai_cli.Tools import Tools
    tools=Tools(client)
    tools.news(news_url)

def search(search_term):
    from gai.cli._gai_cli.Tools import Tools
    tools=Tools(client)
    tools.do_gg(search_term)


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Gai CLI Tool')
    parser.add_argument('command', choices=['init', 'pull', 'news', 'search','docker'], help='Command to run')
    parser.add_argument('-f', '--force', action='store_true', help='Force initialization')
    parser.add_argument('extra_args', nargs='*', help='Additional arguments for commands')

    args = parser.parse_args()

    if args.command == "init":
        print("Initializing...by force" if args.force else "Initializing...")
        init(force=args.force)
    elif args.command == "pull":
        if args.extra_args:
            pull(args.extra_args[0])
        else:
            console.print("[red]Model name not provided[/]")
    elif args.command == "news":
        if args.extra_args:
            news(args.extra_args[0])
        else:
            news()
    elif args.command == "search":
        if args.extra_args:
            search(args.extra_args[0])
        else:
            console.print("[red]Search term not provided[/]")
    elif args.command == "docker":
        if args.extra_args:
            if args.extra_args[0] == "start":
                docker_start()
            elif args.extra_args[0] == "stop":
                docker_stop()
            else:
                console.print("[red]Invalid docker command[/]")


if __name__ == "__main__":
    main()
