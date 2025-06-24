#!/usr/bin/env python3
import os
import re
import sys
import tempfile
import subprocess
from rich import print


def get_version_from_pyproject():
    # locate pyproject.toml one directory up from this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    pyproject_path = os.path.join(script_dir, "..", "pyproject.toml")
    with open(pyproject_path, "r") as f:
        for line in f:
            line = line.strip()
            m = re.match(r'version\s*=\s*"([^"]+)"', line)
            if m:
                return m.group(1)
    sys.exit("❌ Version not found in pyproject.toml")


def smoke_test(use_editable: bool = False):
    version = get_version_from_pyproject()
    print(f"[yellow]🔍 Testing gai-lib version: {version}[/yellow]")

    with tempfile.TemporaryDirectory() as tmpdir:
        # create a temp environment
        env_dir = os.path.join(tmpdir, "env")

        # venv.create(env_dir, with_pip=True)
        env_dir = os.path.join(tmpdir, "env")
        subprocess.check_call(["uv", "venv", env_dir, "--seed"])
        env = os.environ.copy()
        env["PATH"] = os.path.join(env_dir, "bin") + os.pathsep + env["PATH"]
        env["VIRTUAL_ENV"] = env_dir
        env["UV_PROJECT_ENVIRONMENT"] = env_dir
        subprocess.check_call(["which", "python"], env=env)

        py = "python"

        # Check version of gai-lib installed in the environment against the one in pyproject.toml
        if use_editable:
            subprocess.check_call([py, "-m", "pip", "install", "-e", "."], env=env)
        else:
            # find all .whl files in dist/
            dist_dir = "dist"
            for fname in os.listdir(dist_dir):
                if fname.endswith(".whl"):
                    wheel = os.path.join(dist_dir, fname)
                    break
            else:
                raise FileNotFoundError("No .whl found in dist/")
            subprocess.check_call([py, "-m", "pip", "install", wheel], env=env)

        # simpler: grep gai-lib version from pip list
        output = subprocess.check_output(
            f"{py} -m pip list --format=freeze | grep gai-lib",
            shell=True,
            env=env,
            text=True,
        )
        # output is like "gai-lib==1.2.3\n"
        installed_version = output.strip().split("==", 1)[1]
        print(f"[green]✅ Installed gai-lib version: {installed_version}[/green]")

        if installed_version != version:
            print(f"[red]⚠️ Version mismatch! Expected {version}[/red]")

        # import gai.lib
        subprocess.check_call([py, "-c", "import gai.lib"], env=env)
        print("[yellow]✅ Can import gai.lib[/]")

    print("[green]🟢 Smoke test passed[/]")


if __name__ == "__main__":
    smoke_test()
