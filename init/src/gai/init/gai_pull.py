import os
import time
import yaml
import httpx
import json
from pathlib import Path
from urllib.parse import urlparse
from rich.console import Console
from huggingface_hub import snapshot_download

os.environ["HF_HUB_ENABLED_HF_TRANSFER"] = "1"
console = Console()

# Constants

GAIRC = "~/.gairc"

# Get app_dir FROM ~/.gairc


def get_app_dir():
    if not os.path.exists(os.path.expanduser(GAIRC)):
        raise Exception(
            f"Config file {GAIRC} not found. Please run 'gai init' to initialize the configuration."
        )
    with open(os.path.expanduser(GAIRC), "r") as f:
        rc = json.load(f)
    app_dir = os.path.abspath(os.path.expanduser(rc["app_dir"]))
    return app_dir


def get_gai_config():
    app_dir = get_app_dir()
    config_path = os.path.join(app_dir, "gai.yml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Configuration file {config_path} not found. Please run 'gai init' to initialize the configuration."
        )
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


def httpx_download(download_url, output_dir):
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    headers = {}
    file_size = 0

    def get_filename_from_response(response, url):
        # Try to get the filename from the Content-Disposition header
        content_disposition = response.headers.get("Content-Disposition")
        if content_disposition:
            params = dict(
                item.strip().split("=")
                for item in content_disposition.split(";")
                if "=" in item
            )
            filename = params.get("filename", None)
            if filename:
                return filename.strip('"')

        # Fallback: use the last part of the URL path
        return os.path.basename(urlparse(url).path) or "downloaded_file"

    with httpx.Client(follow_redirects=True) as client:
        with client.stream(
            "GET", download_url, headers=headers, timeout=None
        ) as response:
            if response.status_code in (200, 206):
                filename = get_filename_from_response(response, download_url)
                output_path = os.path.join(output_dir, filename)

                # Check if the file already exists to support resuming
                if os.path.exists(output_path):
                    file_size = os.path.getsize(output_path)
                    headers["Range"] = f"bytes={file_size}-"

                total_size = int(response.headers.get("Content-Length", 0)) + file_size
                with open(output_path, "ab") as file:
                    for chunk in response.iter_bytes():
                        file.write(chunk)
                        file_size += len(chunk)
                        # Calculate and print progress percentage
                        progress = (file_size / total_size) * 100
                        console.print(
                            f"Downloaded [italic bright_white]{file_size}[/] of [bold bright_white]{total_size}[/] bytes ([bright_yellow]{progress:.2f}[/]%)",
                            end="\r",
                        )
                        # print(f"\rDownloaded {file_size} of {total_size} bytes ({progress:.2f}%)", end="")
            else:
                print(f"Failed to download. HTTP Status code: {response.status_code}")


def pull(model_name, dry_run=False):
    if not model_name:
        console.print("[red]Model name not provided[/]")
        return False
    app_dir = get_app_dir()
    gai_config = get_gai_config()
    models = gai_config.get("pull", {})
    if not models:
        console.print(
            "[red]No models found in gai.yml. Make sure you have ran the latest gai-init and try again.[/]"
        )
        return False

    model = models.get(model_name, None)
    if not model:
        console.print(
            f"[red]Model {model_name} not found. Make sure you have ran the latest gai-init and try again.[/]"
        )
        return False

    if dry_run:
        console.print(
            f"[yellow]Dry run mode: config for {model_name} is found and ready for download[/]"
        )
        return True

    start = time.time()
    console.print(f"[white]Downloading... {model_name}[/]")
    local_dir = f"{app_dir}/models/" + model["local_dir"]

    if model["type"] == "huggingface":
        if "file" in model:
            snapshot_download(
                repo_id=model["repo_id"],
                local_dir=local_dir,
                revision=model["revision"],
                allow_patterns=model["file"],
            )
        else:
            snapshot_download(
                repo_id=model["repo_id"],
                local_dir=local_dir,
                revision=model["revision"],
            )
    elif model["type"] == "civitai":
        httpx_download(download_url=model["download"], output_dir=local_dir)
    elif model["type"] == "others" and model_name == "coqui-xttsv2":
        import os

        os.environ["COQUI_TOS_AGREED"] = "1"
        from TTS.utils.manage import ModelManager

        mm = ModelManager(output_prefix=local_dir)
        model_name = "tts_models/multilingual/multi-dataset/xtts_v2"
        mm.download_model(model_name)

    end = time.time()
    duration = end - start
    download_size = Path(local_dir).stat().st_size

    from rich.table import Table

    table = Table(title="Download Information")
    # Add columns
    table.add_column("Model Name", justify="left", style="bold yellow")
    table.add_column("Time Taken (s)", justify="right", style="bright_green")
    table.add_column("Size (Mb)", justify="right", style="bright_green")
    table.add_column("Location", justify="right", style="bright_green")

    # Add row with data
    table.add_row(model_name, f"{duration:4}", f"{download_size:2}", local_dir)

    # Print the table to the console
    console.print(table)

    return True
