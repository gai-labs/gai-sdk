import subprocess
import os,sys
import toml

this_dir=os.path.dirname(os.path.realpath(__file__))
from rich.console import Console
# console=Console()
# import toml
# import pkg_resources
# import importlib

# PYTHON = os.path.expanduser("~/.venv/bin/python")

def _cmd(cmd):
    try:
        subprocess.run(cmd, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print("Error: ", e)
        return

def _get_version(pyproject_path):
    data = toml.load(pyproject_path)
    if data.get("project",None):
        version = data['project']['version']
    else:
        version = data["tool"]["poetry"]['version']
    return version

def _get_project_name(pyproject_path):
    data = toml.load(pyproject_path)
    if data.get("project",None):
        version = data['project']['name']
    else:
        version = data["tool"]["poetry"]['name']
    return version

def _update_version(pyproject_path):
    with open(pyproject_path, "r+") as f:
        data = toml.load(f)

        # Extract and update the version number
        version_parts = data["project"]["version"].split(".")
        version_parts[-1] = str(int(version_parts[-1]) + 1)  # Increment the patch version
        data["project"]["version"] = ".".join(version_parts)

        # Write the updated data back to pyproject.toml
        f.seek(0)
        f.write(toml.dumps(data))
        f.truncate()  # Ensure file is truncated if new data is shorter

        return data["project"]["version"]

def _publish_package(proj_path):

    ## Update version in pyproject.toml
    def __update_version():
        # Load the pyproject.toml file
        with open("pyproject.toml", "r+") as f:
            data = toml.load(f)

            # Extract and update the version number
            if data.get("project",None):
                version_parts = data["project"]["version"].split(".")
                version_parts[-1] = str(int(version_parts[-1]) + 1)  # Increment the patch version
                data["project"]["version"] = ".".join(version_parts)
            else:
                version_parts = data["tool"]["poetry"]["version"].split(".")
                version_parts[-1] = str(int(version_parts[-1]) + 1)  # Increment the patch version
                data["tool"]["poetry"]["version"] = ".".join(version_parts)

            # Write the updated data back to pyproject.toml
            f.seek(0)
            f.write(toml.dumps(data))
            f.truncate()  # Ensure file is truncated if new data is shorter

    def __remove_dist_dir():
        # Remove the dist directory
        subprocess.run(["rm", "-rf", "dist"], check=True)

    def __extract_tar_file(tar_file, dest_dir):
        # Ensure the destination directory exists
        os.makedirs(dest_dir, exist_ok=True)
        # Extract the tar file into the destination directory
        subprocess.run(["tar", "xvf", tar_file, "-C", dest_dir], check=True)

    def __build_package():
        # create package distributions
        subprocess.run([PYTHON,"-m", "build","--outdir","dist/"], check=True)

        # Extract all tar.gz files in the dist directory
        # import glob
        # tar_files = glob.glob("dist/*.tar.gz")
        # for tar_file in tar_files:
        #     __extract_tar_file(tar_file, "dist/")

    def __publish_package(proj_path):
        # Upload the package to PyPI
        import glob
        dist_files = glob.glob("dist/*.tar.gz") + glob.glob("dist/*.whl")
        subprocess.run([PYTHON, "-m", "twine", "upload"] + dist_files, check=True)        
        
    # def __publish_package(env, proj_path):
    #     os.system(f"eval \"$(conda shell.bash hook)\" && conda activate {env} && cd {proj_path} && TWINE_USERNAME=__token__ twine upload dist/*")

    print("Updating version in pyproject.toml")
    __update_version()

    print("Building the package")
    __remove_dist_dir()
    __build_package()

    print("Publishing the package to PyPI")
    try:
        __publish_package(proj_path)
        print("Package published successfully.")
    except subprocess.CalledProcessError:
        print("Failed to publish the package.", file=sys.stderr)
        sys.exit(1)

# def _get_package_info(package_name):
#     try:
#         pkg = pkg_resources.get_distribution(package_name)
#         print(pkg)
#         return f"Package: {pkg.project_name}, Version: {pkg.version}"
#     except pkg_resources.DistributionNotFound:
#         return f"Package '{package_name}' is not installed."    

