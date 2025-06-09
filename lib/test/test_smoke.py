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
        venv.create(env_dir, with_pip=True)
        py = os.path.join(env_dir, "bin", "python")

        # install the exact version we just read
        subprocess.check_call([py, "-m", "pip", "install", f"gai-sdk=={version}"])

        # verify import works
        subprocess.check_call([
            py, "-c",
            "import importlib.resources as pkg_resources;print(f'config_path={pkg_resources.path(\"data\", \"gai.yml\")}')"
        ])

    print("🟢 Smoke test passed")

if __name__ == "__main__":
    smoke_test()
