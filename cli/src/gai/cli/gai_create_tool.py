import os
import shutil
from rich.console import Console
console = Console()

TEMPLATE_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__), 
        "..",
        "..",
        "data",
        "data",
        "gai",
        "project-templates",
        "tool-svr"
    )
)
BAR = 40*"─"


def _copy_path(
    template_relative_path_src:str,
    project_dir:str):
    
    """
    
    Example:
    
    _copy_path(".devcontainer/devcontainer.json","/tmp/web")
    
    This copies the contents of ".devcontainer/devcontainer.json" in TEMPLATE_DIR to "/tmp/web/.devcontainer/devcontainer.json".
    
    """
    
    src = os.path.join(TEMPLATE_DIR, template_relative_path_src)
    dst = os.path.join(project_dir, template_relative_path_src)

    # Start copying
    
    if os.path.isfile(src):
        shutil.copyfile(src, dst)
    elif os.path.isdir(src):
        shutil.copytree(src, dst, dirs_exist_ok=True)
    else:
        raise FileNotFoundError(f"Source path {src} does not exist.")
    assert os.path.exists(dst), f"Failed to copy {src} to {dst}"
    console.print(f"[yellow]Copied {src} to {dst}[/]")
    
def _replace_project_name(file_path:str,project_name:str):
    """
    Replace the placeholder {{PROJECT_NAME}} in the file with the actual project name.
    """
    with open(file_path, "r") as f:
        content = f.read()
        content = content.replace("{{PROJECT_NAME}}", project_name)
    with open(file_path, "w") as f:
        f.write(content)
    console.print(f"[yellow]Replaced {{PROJECT_NAME}} with '{project_name}' in {file_path}[/]")

def _replace_tool_name(file_path:str,tool_name:str):
    """
    Replace the placeholder {{PROJECT_NAME}} in the file with the actual project name.
    """
    with open(file_path, "r") as f:
        content = f.read()
        content = content.replace("{{TOOL_NAME}}", tool_name)
    with open(file_path, "w") as f:
        f.write(content)
    console.print(f"[yellow]Replaced {{TOOL_NAME}} with '{tool_name}' in {file_path}[/]")

def get_tool_name(project_name:str):

    # gai-tool-<tool_name>-<others>
    # The 3rd segment is the tool name
    
    segments=project_name.split("-")
    tool_name=segments[2]
    return tool_name

def create_tool(project_name:str):
    project_dir = os.path.join(os.getcwd(), project_name)
    tool_name = get_tool_name(project_name)
    
    console.print(f"[yellow] create tool {project_name} will create project in '{project_dir}' [/]")
    i = input("Do you want to continue? (y/n): ")
    if i.lower() != 'y':
        console.print("[red]Aborted by user.[/]")
        return
    
    # Create project directory
    version_number = "0.0.1"
    os.makedirs(project_dir, exist_ok=True)
    os.chdir(project_dir)
    
    # Create server directory
    def create_server():
        server_dir = os.path.join(project_dir, "server")
        os.makedirs(server_dir, exist_ok=True)
        
        # Create server/_utils
        
        def create_server_utils():
            
            # server/_utils
            
            utils_dir = os.path.join(server_dir, "_utils")
            os.makedirs(utils_dir, exist_ok=True)
            
            # server/_utils/*.py
            
            _copy_path("server/_utils", project_dir)
            
            console.print(f"[green]{BAR} ./server/_utils completed {BAR}[/]")
            
    
        # Create server/.devcontainer dir
        
        def create_server_devcontainer(project_name,tool_name):
            
            # server/.devcontainer
            
            devcontainer_dir = os.path.join(server_dir, ".devcontainer")
            os.makedirs(server_dir, exist_ok=True)

            # server/.devcontainer/library-scripts

            _copy_path("server/.devcontainer/library-scripts", project_dir)
            
            # server/.devcontainer/devcontainer.json

            _copy_path("server/.devcontainer/devcontainer.json", project_dir)
            file_path = os.path.join(devcontainer_dir, "devcontainer.json")
            _replace_project_name(file_path,project_name)
            
            # .devcontainer/docker-compose.yml
            
            _copy_path("server/.devcontainer/docker-compose.yml", project_dir)
            file_path = os.path.join(devcontainer_dir, "docker-compose.yml")
            _replace_project_name(file_path,project_name)
            _replace_tool_name(file_path,tool_name)

            # .devcontainer/Dockerfile.devcontainer

            _copy_path("server/.devcontainer/Dockerfile.devcontainer", project_dir)

            # .devcontainer/postCreateCommand.sh

            _copy_path("server/.devcontainer/postCreateCommand.sh", project_dir)
            
            console.print(f"[green]{BAR} ./server/.devcontainer completed {BAR}[/]")
        
        # Create .vscode dir
        
        def create_server_vscode(project_name,tool_name):
            
            # .vscode
            
            vscode_dir = os.path.join(server_dir, ".vscode")
            os.makedirs(vscode_dir, exist_ok=True)

            # .vscode/settings.json

            _copy_path("server/.vscode/settings.json", project_dir)
            
            # .vscode/launch.json

            _copy_path("server/.vscode/launch.json", project_dir)
            file_path = os.path.join(vscode_dir, "launch.json")
            _replace_project_name(file_path,project_name)
            _replace_tool_name(file_path,tool_name)
            
            console.print(f"[green]{BAR} ./server/.vscode completed {BAR}[/]")
        
        def create_server_gai_lib():
            gai_lib_dir = os.path.join(server_dir, "gai-lib")
            os.makedirs(gai_lib_dir, exist_ok=True)
            console.print(f"[green]{BAR} ./server/gai-lib completed {BAR}[/]")
            
        def create_server_src():
            
            # server/src
            
            src_dir = os.path.join(server_dir, "src")
            os.makedirs(src_dir, exist_ok=True)

            # server/src/gai/tools/<project_name>/client
            
            client_dir = os.path.join(src_dir, "gai", "tools",project_name, "client")
            os.makedirs(client_dir, exist_ok=True)
            
            # server/src/gai/tools/<project_name>/lib
            
            lib_dir = os.path.join(src_dir, "gai", "tools",project_name, "lib")
            os.makedirs(lib_dir, exist_ok=True)
            
            # server/src/gai/tools/<project_name>/svr
            
            svr_dir = os.path.join(src_dir, "gai", "tools",project_name, "svr")
            os.makedirs(svr_dir, exist_ok=True)
            
            console.print(f"[green]{BAR} ./server/src dir completed {BAR}[/]")

        def create_server_test():
            test_dir = os.path.join(server_dir, "test")
            os.makedirs(test_dir, exist_ok=True)
            console.print(f"[green]{BAR} ./server/test dir completed {BAR}[/]")
            
        def create_server_pyproject_toml():
            pyproject_toml_path = os.path.join(server_dir, "pyproject.toml")
            _copy_path("server/pyproject.toml", project_dir)
            _replace_project_name(pyproject_toml_path,project_name)
            _replace_tool_name(pyproject_toml_path,tool_name)
            console.print(f"[green]{BAR} ./server/pyproject.toml completed {BAR}[/]")
            
        def create_server_makefile():
            _copy_path("server/Makefile", project_dir)
            console.print(f"[green]{BAR} ./server/Makefile completed {BAR}[/]")

        create_server_utils()
        create_server_devcontainer(project_name=project_name,tool_name=tool_name)
        create_server_vscode(project_name=project_name,tool_name=tool_name)
        create_server_gai_lib()
        create_server_src()
        create_server_test()
        create_server_pyproject_toml()
        create_server_makefile()
        
        console.print(f"[green]{BAR} ./server dir completed {BAR}[/]")

    def create_src():
        src_dir = os.path.join(project_dir, "src")
        os.makedirs(src_dir, exist_ok=True)
        console.print(f"[green]{BAR} ./src dir completed {BAR}[/]")

    def create_test():
        test_dir = os.path.join(project_dir, "test")
        os.makedirs(test_dir, exist_ok=True)
        console.print(f"[green]{BAR} ./test dir completed {BAR}[/]")

    def create_gitignore():
        _copy_path(".gitignore", project_dir)
    
    def create_pyproject_toml():
        _copy_path("pyproject.toml", project_dir)
        pyproject_toml_path = os.path.join(project_dir,"pyproject.toml")
        _replace_project_name(pyproject_toml_path,project_name)
        _replace_tool_name(pyproject_toml_path,tool_name)
        
    def create_readme():
        readme_path = os.path.join(project_dir,"README.md")
        if not os.path.exists(readme_path):
            with open(readme_path,"w") as f:
                f.write("")

    def create_license():
        _copy_path("LICENSE", project_dir)        

    create_server()
    create_src()
    create_test()
    create_gitignore()
    create_pyproject_toml()
    create_readme()
    create_license()
    
        