import os, sys, re, time
import json
from os.path import dirname
import shutil
from gai.lib.common import constants
import yaml

def init():
    with open(os.path.expanduser(constants.GAIRC), "w") as file:
        file.write(json.dumps({
            "app_dir": "~/.gai"
        }, indent=4))
    config_dir=dirname(dirname(dirname(__file__)))
    config_path=os.path.join(config_dir, 'gai.yml')
    os.makedirs(os.path.expanduser("~/gai/models"), exist_ok=True)
    shutil.copy(config_path, os.path.expanduser("~/gai"))

# Get JSON FROM ~/.gairc
def get_rc():
    if (not os.path.exists(os.path.expanduser(constants.GAIRC))):
        init()
    with open(os.path.expanduser(constants.GAIRC), 'r') as f:
        return json.load(f)

# Get "app_dir" from ~/.gairc
def get_app_path():
    rc = get_rc()
    app_dir=os.path.abspath(os.path.expanduser(rc["app_dir"]))
    return app_dir

def get_gai_config(file_path=None):
    app_dir=get_app_path()
    lib_config_path = os.path.join(app_dir, 'gai.yml')
    if file_path:
        lib_config_path = file_path
    with open(lib_config_path, 'r') as f:
        return yaml.load(f, Loader=yaml.FullLoader)

def this_dir(file):
    return os.path.dirname(os.path.realpath(file))

# Create ~/.gaiaio/cache
def mkdir_cache():
    cache_dir = os.path.expanduser('~/.gai/cache')
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir

def is_url(s):
    return re.match(r'^https?:\/\/.*[\r\n]*', s) is not None

def sha256(text):
    import hashlib
    return hashlib.sha256(text.encode('utf-8')).hexdigest()

def timestamp():
    return int(time.time() * 1000)

def find_url_in_text(text):
    url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    urls = re.findall(url_pattern, text)
    return urls

def clean_string(s):
    if s is None:
        return ''
    return re.sub(r'\s+', ' ', s)

def find_site_packages_path(virtual_env_name):
    # Extracting the Python version
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}"

    # Constructing the path pattern to the virtual environment's site-packages directory
    site_packages_path = os.path.expanduser(f"~/miniconda/envs/{virtual_env_name}/lib/python{python_version}/site-packages")
    return site_packages_path

def find_egg_link(virtual_env_name, package_name):
    site_packages_path = find_site_packages_path(virtual_env_name)
    egg_link_file = os.path.join(site_packages_path, f"{package_name}.egg-link")
    if os.path.exists(egg_link_file):
        return egg_link_file
    else:
        return None

def find_project_path(virtual_env_name, package_name):
    egg_link_file = find_egg_link(virtual_env_name, package_name)
    if egg_link_file is None:
        return None
    else:
        with open(egg_link_file) as f:
            project_path = f.readline().strip()
            return project_path

def free_mem():
    from rich.console import Console
    console = Console()    
    import pynvml
    pynvml.nvmlInit()
    handle=pynvml.nvmlDeviceGetHandleByIndex(0)
    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    free_amt = info.free / 1024**3
    if free_amt < 4:
        console.print(f"Free memory: [bright_red]{free_amt:.2f} GB[/]")
    else:
        console.print(f"Free memory: [bright_green]{free_amt:.2f} GB[/]")
    pynvml.nvmlShutdown()
    return info.free / 1024**3
