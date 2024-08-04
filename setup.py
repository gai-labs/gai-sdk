from setuptools import setup,find_packages
import os

# Is this used for publishing?
is_publishing=os.environ.get("TWINE_USERNAME",False)

# Read the version from the VERSION file
base_dir = os.path.dirname(os.path.abspath(__file__))
version_file = os.path.join(base_dir, 'VERSION')
with open(version_file, 'r') as f:
    VERSION = f.read().strip()
here = os.path.dirname(os.path.realpath(__file__))

# Read the requirements from the requirements.txt file
def parse_requirements(filename):
    if os.path.exists(os.path.join(here, filename)):
        with open(os.path.join(here, filename)) as f:
            required = f.read().splitlines()
        return required

# Install local wheel file
def local_pkg(name: str,file_path: str) -> str:
    """Returns a path to a local package."""
    file_path=os.path.join(os.getcwd(),"wheels",file_path)
    return f"{name} @ file://{file_path}"

extras_require = {}

# For pre-built wheels on localhost (not for docker or publishing)
svr_requirements=[]
if not is_publishing:
    svr_requirements=[local_pkg("exllamav2","exllamav2-0.1.4-cp310-cp310-linux_x86_64.whl")]
extras_require["svr"] = svr_requirements

# For pre-built wheels and development on localhost (not for docker or publishing)
dev_requirements=[]
if dev_requirements and not is_publishing:
    dev_requirements = parse_requirements("requirements_dev.txt")
    dev_requirements.append(local_pkg("exllamav2","exllamav2-0.1.4-cp310-cp310-linux_x86_64.whl"))
    extras_require["dev"] = dev_requirements

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
        parse_requirements("src/gai/ttt/requirements_ttt.txt")
    ],
    extras_require=extras_require,
    entry_points={
        'console_scripts': [
            'gai=gai.cli.scripts.main:main',
        ],
    }
)