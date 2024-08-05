from setuptools import setup,find_packages
import os

# Is this used for publishing?
is_publishing=os.environ.get("TWINE_USERNAME",False)

# Find base directory with reference to setup.py
def get_here():
    return os.path.dirname(os.path.abspath(__file__))
HERE = get_here()
print(f"\033[33mHERE: {HERE}\033[0m")

# Read the version from the VERSION file
def get_version():
    version_file = os.path.join(HERE, 'VERSION')
    with open(version_file, 'r') as f:
        return f.read().strip()
VERSION = get_version()
print(f"\033[33mInstalling: {VERSION}\033[0m")


# Read the requirements from the requirements.txt file
DEFAULT_REQUIREMENTS_PATH=os.path.join(HERE,"src/gai/lib/requirements.txt")
def parse_requirements(filename):
    required=[]
    filepath = os.path.join(HERE, filename)
    if os.path.exists(filepath):
        with open(filepath) as f:
            required = f.read().splitlines()
        return required
DEFAULT_REQUIREMENTS = parse_requirements("./src/gai/lib/requirements.txt")

if not DEFAULT_REQUIREMENTS or len(DEFAULT_REQUIREMENTS)==0:
    raise Exception(f"No requirements found in {DEFAULT_REQUIREMENTS_PATH}")

# Install local wheel file
def local_pkg(name: str,file_path: str) -> str:
    """Returns a path to a local package."""
    file_path=os.path.join(os.getcwd(),"wheels",file_path)
    return f"{name} @ file://{file_path}"

# Merge two lists and remove duplicates
def merge_list(list1, list2):
    # Ensure both list1 and list2 are lists, default to empty list if None
    list1 = list1 if list1 is not None else []
    list2 = list2 if list2 is not None else []
    return list(set(list1) | set(list2))


# For default and cli use
cli_requirements=DEFAULT_REQUIREMENTS.copy()

# For development, debugging and testing on local machine
dev_requirements = merge_list(DEFAULT_REQUIREMENTS,[
    "pytest",
    "ipykernel",
    "pynvml",
    "nats-py",
])

# For running servers with pre-built wheels on localhost (not for docker or publishing)
ttt_requirements = merge_list(DEFAULT_REQUIREMENTS,parse_requirements("./src/gai/ttt/requirements_ttt.txt"))
ttt_requirements.append(local_pkg("exllamav2","exllamav2-0.1.4-cp310-cp310-linux_x86_64.whl"))

# For nats
net_requirements = merge_list(DEFAULT_REQUIREMENTS,parse_requirements("./src/gai/net/server/requirements.txt"))

setup(
    name='gai-sdk',
    version=VERSION,
    author="kakkoii1337",
    author_email="kakkoii1337@gmail.com",
    package_dir={'': 'src'},
    packages=find_packages(where='src'),
    description = "Refer to https://gai-labs.github.io/gai for more information",
    long_description="Refer to https://gai-labs.github.io/gai for more information",
    long_description_content_type="text/markdown",
    classifiers=[
        'Programming Language :: Python :: 3.10',
        "Development Status :: 3 - Alpha",        
        'License :: OSI Approved :: MIT License',
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",        
        'Operating System :: OS Independent',
        "Topic :: Software Development",
        "Topic :: Software Development :: Libraries :: Python Modules",        
        "Topic :: Scientific/Engineering :: Artificial Intelligence",        
    ],
    python_requires='>=3.10',
    install_requires=[
        DEFAULT_REQUIREMENTS
    ],
    extras_require= {
        "cli": cli_requirements,
        "dev": dev_requirements,
        "ttt": ttt_requirements,
        "net": net_requirements,
    } if not is_publishing else {},
    entry_points={
        'console_scripts': [
            'gai=gai.cli.scripts.main:main',
        ],
    }
)