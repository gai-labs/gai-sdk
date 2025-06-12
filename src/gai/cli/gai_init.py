from pathlib import Path
from rich.console import Console
import shutil
import json,os
import importlib.resources as pkg_resources

PACKAGED_DATA_PATH = pkg_resources.path('data', '')
PACKAGED_CONFIG_PATH = pkg_resources.path('data', 'gai.yml')


def init_gairc(force):
    if Path("~/.gairc").expanduser().exists():
        if not force:
            print("~/.gairc exists.")
            return
        else:
            print("Deleting existing ~/.gairc")
            os.remove(Path("~/.gairc").expanduser())

    # Create .gairc if not already exists OR force=True
    print("~/.gairc")    
    with open(Path("~/.gairc").expanduser(), "w") as f:
        f.write(json.dumps({
            "app_dir":"~/.gai"
        }))
        
def init_gai_dir(force):
    if Path("~/.gai").expanduser().exists():
        if not force:
            print("~/.gai exists.")
            return
        else:
            print("Deleting existing ~/.gai")
            shutil.rmtree(Path("~/.gai").expanduser())
    # Create .gai if not already exists OR force=True
    print("Creating ~/.gai")
    Path("~/.gai").expanduser().mkdir()

def init_gai_models_dir(force):
    if Path("~/.gai/models").expanduser().exists():
        if not force:
            print("~/.gai/models exists.")
            return
        else:
            print("Deleting existing ~/.gai/models")
            shutil.rmtree(Path("~/.gai/models").expanduser())
    
    # Create .gai/models if not already exists OR force=True
    print("Creating ~/.gai/models")
    
    # Create the directory for some of the common models as mount points before the services creates them using "root" user.
    Path("~/.gai/models").expanduser().mkdir()  
    Path("~/.gai/models/ollama").expanduser().mkdir()  
    Path("~/.gai/models/Stable-diffusion").expanduser().mkdir()  
    Path("~/.gai/models/VAE").expanduser().mkdir()
    Path("~/.gai/models/instructor-large").expanduser().mkdir()

def init_gai_projects_dir(force):
    if Path("~/.gai/projects").expanduser().exists():
        if not force:
            print("~/.gai/projects exists.")
            return
        else:
            print("Deleting existing ~/.gai/projects")
            shutil.rmtree(Path("~/.gai/projects").expanduser())
    
    # Create .gai/models if not already exists OR force=True
    print("Creating ~/.gai/projects")

def copy_gai_yml(force):
    if Path("~/.gai/gai.yml").expanduser().exists():
        if not force:
            print("gai.yml exists.")
            return
        else:
            print("Deleting existing gai.yml")
            shutil.rmtree(Path("~/.gai/gai.yml").expanduser())
    # Create gai.yml if not already exists OR force=True
    print("Copying gai.yml")
    shutil.copy(PACKAGED_CONFIG_PATH, Path("~/.gai/gai.yml").expanduser())
    
    
def copy_package_data(force):
    
    if Path("~/.gai/data").expanduser().exists():
        if not force:
            print("~/.gai/data exists.")
            return
        else:
            print("Deleting existing ~/.gai/data")
            shutil.rmtree(Path("~/.gai/data").expanduser())
            
    print("Copying ~/.gai/data")
    # Copy all files except gai.yml in PACKAGED_DATA_PATH to ~/.gai
    DESTINATION = Path("~/.gai").expanduser()
    ignore_patterns = shutil.ignore_patterns('__pycache__', '__init__.py')     
    for item in PACKAGED_DATA_PATH.glob("*"):
        if item.is_dir():
            shutil.copytree(
                item,
                DESTINATION / item.name,
                ignore=ignore_patterns
            )
        else:
            # Skip __init__.py files in the top-level
            if item.name != '__init__.py':
                shutil.copy(item, DESTINATION / item.name)      

def init(force=False):
    console=Console()

    # Initialise config
    init_gairc(force)
    init_gai_dir(force)
    init_gai_models_dir(force)
    init_gai_projects_dir(force)
    copy_gai_yml(force)
    copy_package_data(force)
    console.print("[green]Initialized[/]")
    
if __name__=="__main__":
    init(True)