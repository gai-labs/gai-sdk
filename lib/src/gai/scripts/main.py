#!/bin/env python3
from pathlib import Path
import json
import sys,os
from openai import OpenAI
from rich.console import Console
console=Console()

here = os.path.abspath(os.path.dirname(__file__))

def app_dir():
    with open(Path("~/.gairc").expanduser(), "r") as file:
        rc=file.read()
        jsoned = json.loads(rc)
        return Path(jsoned["app_dir"]).expanduser()

GENERAL_USAGE_HINT = """
[yellow]Usage: gai <command> {options} {args}
Commands:
  init        Initialize the GAI environment
  pull        Pull a model from the GAI repository
  create      Create a new project (tool, agent, or model)
[/]
"""

CREATE_USAGE_HINT = """
[yellow]Usage: gai create (tool|agent|model) <name>
Types:
  tool       Create a new tool project
  agent      Create a new agent project
  model      Create a new model project
[/]
"""

def main():
    import argparse

    parser = argparse.ArgumentParser(description='Gai CLI Tool')
    parser.add_argument('command', choices=['init', 'pull','create'], help='Command to run')
    parser.add_argument('-f', '--force', action='store_true', help='Force initialization')
    parser.add_argument('extra_args', nargs='*', help='Additional arguments for commands')
    parser.add_argument("--repo-name", default="kakkoii1337", help="Repository name for Docker image.")
    parser.add_argument("--image-name", help="Base name for Docker image, which is required.")
    parser.add_argument("--dockerfile-path", default="./Dockerfile", help="Path to the Dockerfile used for building the image.")
    parser.add_argument("--dockercontext-path", default=".", help="Path to the Dockerfile used for building the image.")
    parser.add_argument("--no-cache", action="store_true", help="Do not use cache when building the image.")

    try:
        args = parser.parse_args()
    except SystemExit:
        console.print("[red]Syntax Error: Invalid command. Use 'init', 'pull', or 'create'[/]")
        console.print(GENERAL_USAGE_HINT)
        raise
    except Exception as e:
        console.print(f"[red]An error occurred: {e}[/]")
        console.print(GENERAL_USAGE_HINT)
        raise

    if args.command == "init":
        from gai.scripts.gai_init import init
        print("Initializing...by force" if args.force else "Initializing...")
        init(force=args.force)
    elif args.command == "pull":
        if args.extra_args:
            from gai.scripts.gai_pull import pull
            pull(console, args.extra_args[0])
        else:
            console.print("[red]Model name not provided[/]")
    elif args.command == "create":
        if args.extra_args:
            if len(args.extra_args) != 2:
                console.print("[red]Syntax Error: Invalid argument count.[/]")
                console.print(CREATE_USAGE_HINT)
                sys.exit(1)
            elif args.extra_args[0] not in ["tool", "agent", "model"]:
                console.print("[red]Syntax Error: Invalid create type. Use 'tool', 'agent' or 'model'. [/]")
                console.print(CREATE_USAGE_HINT)
                sys.exit(1)
            project_type = args.extra_args[0]
            project_name = args.extra_args[1]
            if project_type == "tool":
                from gai.scripts.gai_create_tool import create_tool
                create_tool(project_name)
            else:
                print(f"The function 'create {project_type}' is not available yet.")
    else:
        console.print("[red]Invalid command[/]")

if __name__ == "__main__":
    main()
