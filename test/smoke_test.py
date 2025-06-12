#!/usr/bin/env python3
import os
import re
import sys
import tempfile
import subprocess
import venv


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


def smoke_test():
    version = get_version_from_pyproject()
    print(f"🔍 Testing gai-sdk version: {version}")

    with tempfile.TemporaryDirectory() as tmpdir:
        env_dir = os.path.join(tmpdir, "env")

        # Create virtual environment using uv
        subprocess.check_call(["uv", "venv", env_dir])

        # Get the python executable path (different for Windows vs Unix)
        if os.name == "nt":  # Windows
            py = os.path.join(env_dir, "Scripts", "python")
        else:  # Unix/Linux/macOS
            py = os.path.join(env_dir, "bin", "python")

        # install the exact version we just read using uv
        subprocess.check_call(
            ["uv", "pip", "install", f"gai-sdk=={version}", "--python", py]
        )

        # verify import works
        subprocess.check_call(
            [
                py,
                "-c",
                'import importlib.resources as pkg_resources;print(f\'✅ config_path={pkg_resources.path("data", "gai.yml")}\')',
            ]
        )

        # verify gai.lib import works
        print("🔍 Testing gai.lib import...")
        subprocess.check_call(
            [
                py,
                "-c",
                'import gai.lib; print("✅ gai.lib imported successfully")',
            ]
        )

        # verify gai.llm import works
        print("🔍 Testing gai.llm import...")
        subprocess.check_call(
            [
                py,
                "-c",
                'import gai.llm; print("✅ gai.llm imported successfully")',
            ]
        )

        # verify gai.mcp import works
        print("🔍 Testing gai.mcp import...")
        subprocess.check_call(
            [
                py,
                "-c",
                'import gai.mcp; print("✅ gai.mcp imported successfully")',
            ]
        )

        # verify gai init works
        print("🔍 Testing gai init...")
        subprocess.check_call([py, "-m", "gai.cli.gai_main", "init"])

    print("🟢 Smoke test passed")


if __name__ == "__main__":
    smoke_test()
