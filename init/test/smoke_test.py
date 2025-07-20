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
    os.chdir("..")  # ⬅ Immediately change directory to parent
    version = get_version_from_pyproject()
    print(f"[yellow]🔍 Testing gai-init version: {version}[/yellow]")

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
            dist_dir = "dist"
            subprocess.check_call(["rm", "-rf", dist_dir])
            subprocess.check_call(
                [py, "-m", "pip", "install", "--upgrade", "build"], env=env
            )
            subprocess.check_call([py, "-m", "build"], env=env)

            # find all .whl files in dist/
            for fname in os.listdir(dist_dir):
                if fname.endswith(".whl"):
                    wheel = os.path.join(dist_dir, fname)
                    break
            else:
                raise FileNotFoundError("No .whl found in dist/")
            subprocess.check_call([py, "-m", "pip", "install", wheel], env=env)

        # simpler: grep gai-init version from pip list
        output = subprocess.check_output(
            f"{py} -m pip list --format=freeze | grep gai-init",
            shell=True,
            env=env,
            text=True,
        )
        # output is like "gai-init==1.2.3\n"
        installed_version = output.strip().split("==", 1)[1]
        print(f"[green]✅ Installed gai-init version: {installed_version}[/green]")

        if installed_version != version:
            print(f"[red]⚠️ Version mismatch! Expected {version}[/red]")

        # Check that `data` directory is installed
        try:
            result = subprocess.check_output(
                [py, "-c", "import importlib.resources as r; print(r.files('data'))"],
                env=env,
                text=True,
                stderr=subprocess.STDOUT,
            )
            print(f"[green]📦 'data' directory found: {result.strip()}[/green]")

            # Check for existence of data/gai.yml inside the installed package
            file_check = subprocess.check_output(
                [
                    py,
                    "-c",
                    (
                        "import importlib.resources as r; "
                        "print(r.files('data').joinpath('gai.yml').is_file())"
                    ),
                ],
                env=env,
                text=True,
                stderr=subprocess.STDOUT,
            )
            if file_check.strip() == "True":
                print("[green]📄 'data/gai.yml' exists in installed package[/green]")
            else:
                print("[red]❌ 'data/gai.yml' not found in installed package[/red]")

        except subprocess.CalledProcessError as e:
            print(
                f"[red]❌ 'data' directory not found or not accessible as a package[/red]"
            )
            print(e.output)

    print("[green]🟢 Smoke test passed[/]")


if __name__ == "__main__":
    smoke_test()
