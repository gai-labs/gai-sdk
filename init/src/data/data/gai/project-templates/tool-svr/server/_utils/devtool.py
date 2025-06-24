#!/usr/bin/env python3
"""
Internal developer utilities for gai-sdk.

Usage examples
--------------
python -m gai.scripts.devtool bump-version         # patch++
python -m gai.scripts.devtool build
python -m gai.scripts.devtool test-dist
python -m gai.scripts.devtool publish
"""
import argparse, pathlib, shutil, subprocess, sys, textwrap, toml, tempfile
from importlib.resources import files
from ._docker_utils import _docker_build, _docker_push

ROOT = pathlib.Path(__file__).resolve().parents[3]   # project root

# ── helpers ───────────────────────────────────────────────────────────────
def bump_version(part: str = "patch"):
    pj = ROOT / "pyproject.toml"
    data = toml.loads(pj.read_text())
    ver = data["project"]["version"].split(".")
    if part == "major":
        ver[0] = str(int(ver[0]) + 1); ver[1:] = ["0", "0"]
    elif part == "minor":
        ver[1] = str(int(ver[1]) + 1); ver[2] = "0"
    else:
        ver[2] = str(int(ver[2]) + 1)
    data["project"]["version"] = ".".join(ver)
    pj.write_text(toml.dumps(data))
    print("📌 bumped to", data["project"]["version"])
    
def inspect_pkg_data():
    import sys, zipfile, pathlib
    wheels = list(pathlib.Path("dist").glob("*.whl"))
    if not wheels:
        print("❌  No wheel found in ./dist – run `gai-dev build` first.")
        sys.exit(1)
    whl = wheels[0]
    with zipfile.ZipFile(whl) as z:
        for name in z.namelist():
            if name.startswith("gai/scripts/data/"):
                print(" •", name.removeprefix("gai/scripts/data/"))

def build():
    shutil.rmtree("dist", ignore_errors=True)
    subprocess.check_call([sys.executable, "-m", "build", "-w", "-s"])

def smoke_install():
    wheel = next(pathlib.Path("dist").glob("*.whl"))
    tmp_venv = pathlib.Path(tempfile.mkdtemp(prefix="gai-test-"))
    subprocess.check_call([sys.executable, "-m", "venv", tmp_venv])
    pip = tmp_venv / "bin" / "pip"
    subprocess.check_call([pip, "install", wheel, "--no-deps"])
    # Quick data-file count
    cnt = sum(1 for _ in files("gai.scripts.data").rglob("*"))
    print("✅ wheel installs, data files:", cnt)

def publish():
    subprocess.check_call(["twine", "upload", "dist/*"])

def docker_build(
    pyproject_path="pyproject.toml",
    repo_name='kakkoii1337',
    image_name=None,                
    dockerfile_path=None, 
    dockercontext_path=None, 
    no_cache=False
):
    """Build a Docker image from a Dockerfile.
    
    Args:
        pyproject_path: Path to the pyproject.toml file
        repo_name: Docker repository name
        image_name: Docker image name (defaults to project name from pyproject.toml)
        dockerfile_path: Path to the Dockerfile (defaults to ./Dockerfile)
        dockercontext_path: Path to the Docker build context (defaults to directory containing pyproject.toml)
        no_cache: If True, don't use cache when building the image
    """
    _docker_build(
        pyproject_path=pyproject_path,
        repo_name=repo_name,
        image_name=image_name, 
        dockerfile_path=dockerfile_path,
        dockercontext_path=dockercontext_path,
        no_cache=no_cache)
    
    
def docker_push(
    pyproject_path="pyproject.toml",
    repo_name='kakkoii1337',
    image_name=None
):
    """Push a Docker image to a repository.
    
    Args:
        pyproject_path: Path to the pyproject.toml file
        repo_name: Docker repository name
        image_name: Docker image name (defaults to project name from pyproject.toml)
    """    
    _docker_push(
        pyproject_path=pyproject_path,
        repo_name=repo_name,
        image_name=image_name)

# ── CLI ───────────────────────────────────────────────────────────────────
def _cli():
    
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("bump-version").add_argument("--part", choices=["major","minor","patch"], default="patch")
    sub.add_parser("build")
    sub.add_parser("test-dist")
    sub.add_parser("publish")
    sub.add_parser("inspect-pkg-data")

    # Add docker_build command
    docker_parser = sub.add_parser("docker_build")
    docker_parser.add_argument("--pyproject-path", default="pyproject.toml", help="Path to pyproject.toml file")
    docker_parser.add_argument("--repo-name", default="kakkoii1337", help="Docker repository name")
    docker_parser.add_argument("--image-name", help="Docker image name")
    docker_parser.add_argument("--dockerfile-path", help="Path to Dockerfile")
    docker_parser.add_argument("--dockercontext-path", help="Path to docker context")
    docker_parser.add_argument("--no-cache", action="store_true", help="Do not use cache when building")
    
    # Add docker_push command
    docker_push_parser = sub.add_parser("docker_push")
    docker_push_parser.add_argument("--pyproject-path", default="pyproject.toml", help="Path to pyproject.toml file")
    docker_push_parser.add_argument("--repo-name", default="kakkoii1337", help="Docker repository name")
    docker_push_parser.add_argument("--image-name", help="Docker image name")
    
    args = p.parse_args()

    if args.cmd == "bump-version": bump_version(args.part)
    elif args.cmd == "build":      build()
    elif args.cmd == "test-dist":  build(); smoke_install()
    elif args.cmd == "inspect-pkg-data": inspect_pkg_data()
    elif args.cmd == "publish":    publish()
    elif args.cmd == "docker_build": 
        docker_build(
            pyproject_path=args.pyproject_path,
            repo_name=args.repo_name,
            image_name=args.image_name,
            dockerfile_path=args.dockerfile_path,
            dockercontext_path=args.dockercontext_path,
            no_cache=args.no_cache
        )
    elif args.cmd == "docker_push":
        docker_push(
            pyproject_path=args.pyproject_path,
            repo_name=args.repo_name,
            image_name=args.image_name
        )        

if __name__ == "__main__": _cli()
