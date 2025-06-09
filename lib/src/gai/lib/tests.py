import os
import sys
import json
import toml
import shutil
import tempfile
import subprocess

def get_local_tmp(request=None):

    """
    Construct path for the test file based on the caller's directory.
    """

    if request is None:
        # If request is not provided, use the current working directory
        caller_dir = os.getcwd()   # ✅ current working directory
        tmp_dir = os.path.join(caller_dir, "tmp")
    else:
        caller_file = request.module.__file__   # ✅ test script file
        caller_dir = os.path.dirname(os.path.abspath(caller_file))
        tmp_dir = os.path.join(caller_dir, "tmp", request.node.name)
    return tmp_dir

def make_local_tmp(request=None):

    """
    Create a tmp/ folder using get_local_tmp()
    """
    
    tmp_dir = get_local_tmp(request)
    shutil.rmtree(tmp_dir, ignore_errors=True)  # Remove the directory if it exists
    os.makedirs(tmp_dir, exist_ok=True)
    return tmp_dir

def get_local_datadir(request=None):

    """
    Create a tmp/ folder inside the directory of the test file that uses this fixture.
    """
    
    if request:
        caller_file = request.module.__file__   # ✅ test script file
        caller_dir = os.path.dirname(os.path.abspath(caller_file))
    else:
        # If request is not provided, use the current working directory
        caller_dir = os.getcwd()

    datadir = os.path.join(caller_dir, "data")

    return datadir


def get_pyproject_path():
    # locate pyproject.toml by traversing up the directory tree
    current_dir = os.getcwd()
    while current_dir != os.path.dirname(current_dir):
        pyproject_path = os.path.join(current_dir, "pyproject.toml")
        if os.path.exists(pyproject_path):
            return pyproject_path
        current_dir = os.path.dirname(current_dir)
    sys.exit("❌ pyproject.toml not found in the directory tree")
    
def get_pyproject_version(pyproject_path):
    # locate pyproject.toml one directory up from this script
    with open(pyproject_path, "r") as f:
        pyproject = toml.load(f)
        version = pyproject.get("project", {}).get("version")
        if version:
            return version
    sys.exit("❌ Version not found in pyproject.toml")

def get_pyproject_name(pyproject_path):
    # locate pyproject.toml one directory up from this script
    with open(pyproject_path, "r") as f:
        pyproject = toml.load(f)
        name = pyproject.get("project", {}).get("name")
        if name:
            return name
    sys.exit("❌ Name not found in pyproject.toml")

def _run_pip_list(env_dir,env):
    python_bin = os.path.join(env_dir, "bin", "python")
    subprocess.check_call([
        python_bin, "-c",
        (
            "import json, subprocess, sys; "
            "data = json.loads(subprocess.check_output([sys.executable, '-m', 'pip', 'list', '--format=json'])); "
            "mods = [pkg for pkg in data if pkg['name'].startswith('gai.') ]; "
            "print('🔍 Installed gai modules:', mods)"
        )
    ], env=env)    

# def _run_uv_pip_list(env_dir,env):
#     subprocess.check_call([
#         "uv", "pip", "list", "--format=json"
#     ], env=env)    
    
def _run_uv_pip_list(env_dir, env):
    import json

    result = subprocess.run(
        ["uv", "pip", "list", "--format=json"],
        env=env,
        capture_output=True,
        text=True,
        check=True
    )

    packages = json.loads(result.stdout)
    gai_packages = [pkg for pkg in packages if pkg["name"].startswith("gai")]

    print("🔍 Installed gai modules:")
    for pkg in gai_packages:
        print(f"  - {pkg['name']} (version: {pkg['version']}) @ {pkg['editable_project_location']}")

def create_temp_env():
    tmpdir = tempfile.TemporaryDirectory()
    env_dir = os.path.join(tmpdir.name, "env")

    subprocess.check_call(["uv", "venv", env_dir])

    env = os.environ.copy()
    env["UV_PROJECT_ENVIRONMENT"] = env_dir

    python_bin = os.path.join(env_dir, "bin", "python")
    subprocess.check_call([python_bin, "-m", "ensurepip", "--upgrade"], env=env)
    subprocess.check_call([python_bin, "-m", "pip", "install", "--upgrade", "pip"], env=env)

    return env_dir, env, tmpdir

def install_package_test(pyproject_path, env_dir, env):
    version = get_pyproject_version(pyproject_path)
    name = get_pyproject_name(pyproject_path)
    print(f"🔍 Testing editable install {name}({version}).")

    project_dir = os.path.dirname(pyproject_path)

    # Install the package in editable mode
    subprocess.check_call([
        "uv", "pip", "install", "-e", f"{project_dir}[dev]"
    ], env=env)

    # Verify import works
    _run_uv_pip_list(env_dir, env)

    print("🟢 Install package test passed")
    
def get_sys_path(env_dir, env):
    import tempfile

    python_bin = os.path.join(env_dir, "bin", "python")

    script = "import sys, json; print(json.dumps(sys.path))"

    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(script)
        script_path = f.name

    try:
        result = subprocess.check_output(
            ["uv", "run", "--", "python", script_path],
            env=env,
            text=True
        )
        return json.loads(result)
    finally:
        os.unlink(script_path)
