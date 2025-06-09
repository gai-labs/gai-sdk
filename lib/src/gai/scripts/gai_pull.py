# requires pip install huggingface-hub
import os
from huggingface_hub import snapshot_download

from gai.lib.utils import get_app_path
from pathlib import Path

os.environ["HF_HUB_ENABLED_HF_TRANSFER"]="1"
import time
import httpx

from rich.console import Console
console=Console()

llm_models = {
    "instructor":{
        "type":"huggingface",
        "repo_id":"hkunlp/instructor-large",
        "local_dir":"instructor-large",
        "revision":"54e5ffb8d484de506e59443b07dc819fb15c7233"
    },
    "llama3.2:3b:exl2": {
        "type":"huggingface",
        "repo_id":"bartowski/dolphin-2.9-llama3-8b-256k-exl2",
        "local_dir":"exl2-llama3_dolphin",
        "revision":"e2e2998baa533c94874995c117115eb70c7a89bb"        
    },
    "dolphin2.9_llama3:8b:exl2": {
        "type":"huggingface",
        "repo_id":"bartowski/dolphin-2.9-llama3-8b-256k-exl2",
        "local_dir":"exl2-llama3_dolphin",
        "revision":"e2e2998baa533c94874995c117115eb70c7a89bb"        
    },
    "dolphin2.8_mistral:7b:exl2":{
        "type":"huggingface",
        "repo_id":"bartowski/dolphin-2.8-mistral-7b-v02-exl2",
        "local_dir":"exllamav2-dolphin",
        "revision":"e2e2998baa533c94874995c117115eb70c7a89bb"
    },    
    "exllamav2-dolphin":{
        # obsolete: Replaced by "exl2-mistral7b_dolphin"
        "type":"huggingface",
        "repo_id":"bartowski/dolphin-2.8-mistral-7b-v02-exl2",
        "local_dir":"exllamav2-dolphin",
        "revision":"e2e2998baa533c94874995c117115eb70c7a89bb"
    },
    "exllamav2-mistral7b":{
        # obsolete: Replaced by "exllamav2-dolphin"
        "type":"huggingface",
        "repo_id":"bartowski/Mistral-7B-Instruct-v0.3-exl2",
        "local_dir":"exllamav2-mistral7b",
        "revision":"1a09a351a5fb5a356102bfca2d26507cdab11111"
    },
    "llamacpp-mistral7b":{
        # obsolete: Use ollama instead
        "type":"huggingface",
        "repo_id":"MaziyarPanahi/Mistral-7B-Instruct-v0.3-GGUF",
        "local_dir":"llamacpp-mistral7b",
        "revision":"ce89f595755a4bf2e2e05d155cc43cb847c78978",
        "file": "Mistral-7B-Instruct-v0.3.Q4_K_M.gguf"
    },
    "llamacpp-dolphin":{
        # obsolete: Use ollama instead
        "type":"huggingface",
        "repo_id":"bartowski/dolphin-2.9.3-mistral-7B-32k-GGUF",
        "local_dir":"llamacpp-dolphin",
        "revision":"740ce4567b3392bd065637d2ac29127ca417cc45",
        "file": "dolphin-2.9.3-mistral-7B-32k-Q4_K_M.gguf"
    },
    "runwayml":{
        "type":"huggingface",
        "repo_id":"runwayml/stable-diffusion-v1-5",
        "revision":"1d0c4ebf6ff58a5caecab40fa1406526bca4b5b9",
        "local_dir":"Stable-diffusion/runwayml"
    },
    "dreamshaper_1.5-civitai":{
        "type":"civitai",
        "url":"https://civitai.com/models/4384/dreamshaper",
        "download":"https://civitai.com/api/download/models/128713?type=Model&format=SafeTensor&size=pruned&fp=fp16",
        "local_dir":"Stable-diffusion/dreamshaper_1.5-civitai",
    },
    "dreamshaper_XL-civitai":{
        "type":"civitai",
        "url":"https://civitai.com/models/112902",
        "download":"https://civitai.com/api/download/models/354657?type=Model&format=SafeTensor&size=full&fp=fp16",
        "local_dir":"Stable-diffusion/dreamshaper_XL-civitai",
    },
    "juggernaut_rb-civitai":{
        "type":"civitai",
        "url":"https://civitai.com/models/46422/juggernaut",
        "download":"https://civitai.com/api/download/models/274039?type=Model&format=SafeTensor&size=pruned&fp=fp16",
        "local_dir":"Stable-diffusion/juggernaut_rb-civitai",
    },
    "absolute_reality-civitai":{
        "type":"civitai",
        "url":"https://civitai.com/models/81458",
        "download":"https://civitai.com/api/download/models/132760?type=Model&format=SafeTensor&size=pruned&fp=fp16",
        "local_dir":"Stable-diffusion/absolute_reality-civitai",
    },
    "icbinp-civitai":{
        "type":"civitai",
        "url":"https://civitai.com/models/28059",
        "download":"https://civitai.com/api/download/models/667760?type=Model&format=SafeTensor&size=pruned&fp=fp16",
        "local_dir":"Stable-diffusion/icbinp-civitai",
    },
    "realistic_vision-civitai":{
        "type":"civitai",
        "url":"https://civitai.com/models/4201",
        "download":"https://civitai.com/api/download/models/501240?type=Model&format=SafeTensor&size=pruned&fp=fp16",
        "local_dir":"Stable-diffusion/realistic_vision-civitai",
    },
    "coqui-xttsv2":{
        "type":"others",
        "local_dir":"coqui-xttsv2",
    },
    "kokoro":{
        "type":"huggingface",
        "repo_id":"hexgrad/Kokoro-82M",
        "local_dir":"kokoro",
        "revision":"e78b910980f63ec856f07ba02a24752a5ab7af5b"
    },
    "whisperv3-huggingface":{
        "type":"huggingface",
        "repo_id":"openai/whisper-large-v3",
        "local_dir":"whisperv3-huggingface",
        "revision":"06f233fe06e710322aca913c1bc4249a0d71fce1"
    },
    "exllamav2-deepseek":{
        "type":"huggingface",
        "repo_id":"bartowski/deepseek-coder-6.7b-instruct-exl2",
        "local_dir":"exllamav2-deepseek",
        "revision":"53bfa0459ca092ab4d206111be453eae148ff5a4"
    },
    "clip-openai":{
        "type":"huggingface",
        "repo_id":"openai/clip-vit-large-patch14",
        "local_dir":"clip-vit-large-patch14",
        "revision":"32bd64288804d66eefd0ccbe215aa642df71cc41"
    },
    "llava1.5-haotian":{
        "type":"huggingface",
        "repo_id":"liuhaotian/llava-v1.5-7b",
        "local_dir":"llava-v1.5-7b",
        "revision":"4481d270cc22fd5c4d1bb5df129622006ccd9234"
    },
    "llava1.5-hf":{
        "type":"huggingface",
        "repo_id":"llava-hf/llava-1.5-7b-hf",
        "local_dir":"llava-1.5-7b-hf",
        "revision":"37a8553f98a8b741b2cf90c8d65753ead1d6c74a"
    },
    "llava1.6-mistral":{
        "type":"huggingface",
        "repo_id":"liuhaotian/llava-v1.6-mistral-7b",
        "local_dir":"llava-v1.6-mistral-7b",
        "revision":"f13b6254afb9d96a82e6f568d7a01101923b3ed9"
    },
    "llava1.6-vicuna":{
        "type":"huggingface",
        "repo_id":"liuhaotian/llava-v1.6-vicuna-7b",
        "local_dir":"llava-v1.6-vicuna-7b",
        "revision":"deae57a8c0ccb0da4c2661cc1891cc9d06503d11"
    },
    "llava1.6-mistral-hf":{
        "type":"huggingface",
        "repo_id":"liuhaotian/llava-v1.6-mistral-7b",
        "":"llava-v1.6-mistral-7b",
        "revision":"75e686c43a9492f588490392b20fa7ac84aa57a7"
    }
}

from urllib.parse import urlparse

def httpx_download(download_url, output_dir):
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    headers = {}
    file_size = 0
    
    def get_filename_from_response(response, url):
        # Try to get the filename from the Content-Disposition header
        content_disposition = response.headers.get("Content-Disposition")
        if content_disposition:
            params = dict(item.strip().split('=') for item in content_disposition.split(';') if '=' in item)
            filename = params.get('filename', None)
            if filename:
                return filename.strip('"')
        
        # Fallback: use the last part of the URL path
        return os.path.basename(urlparse(url).path) or "downloaded_file"


    with httpx.Client(follow_redirects=True) as client:
        with client.stream("GET", download_url, headers=headers, timeout=None) as response:
            if response.status_code in (200, 206):
                filename = get_filename_from_response(response, download_url)
                output_path = os.path.join(output_dir, filename)

                # Check if the file already exists to support resuming
                if os.path.exists(output_path):
                    file_size = os.path.getsize(output_path)
                    headers['Range'] = f'bytes={file_size}-'

                total_size = int(response.headers.get('Content-Length', 0)) + file_size
                with open(output_path, 'ab') as file:
                    for chunk in response.iter_bytes():
                        file.write(chunk)
                        file_size += len(chunk)
                        # Calculate and print progress percentage
                        progress = (file_size / total_size) * 100                        
                        console.print(f"Downloaded [italic bright_white]{file_size}[/] of [bold bright_white]{total_size}[/] bytes ([bright_yellow]{progress:.2f}[/]%)", end="\r")
                        #print(f"\rDownloaded {file_size} of {total_size} bytes ({progress:.2f}%)", end="")
            else:
                print(f"Failed to download. HTTP Status code: {response.status_code}")


def pull(console, model_name):
    app_dir = get_app_path()
    if not model_name:
        console.print("[red]Model name not provided[/]")
        return
    model=llm_models.get(model_name,None)
    if not model:
        console.print(f"[red]Model {model_name} not found[/]")
        return

    start=time.time()
    console.print(f"[white]Downloading... {model_name}[/]")
    local_dir=f"{app_dir}/models/"+model["local_dir"]
    
    if model["type"]=="huggingface":
        
        if "file" in model:
            snapshot_download(
                repo_id=model["repo_id"],
                local_dir=local_dir,
                revision=model["revision"],
                allow_patterns=model["file"]
                )
        else:
            snapshot_download(
                repo_id=model["repo_id"],
                local_dir=local_dir,
                revision=model["revision"],
                )
    elif model["type"]=="civitai":
        httpx_download(
            download_url=model["download"],
            output_dir=local_dir
        )
    elif model["type"]=="others" and model_name=="coqui-xttsv2":
        import os
        os.environ["COQUI_TOS_AGREED"]="1"
        from TTS.utils.manage import ModelManager
        mm =  ModelManager(output_prefix=local_dir)
        model_name="tts_models/multilingual/multi-dataset/xtts_v2"
        mm.download_model(model_name)
        
    end=time.time()
    duration=end-start
    download_size=Path(local_dir).stat().st_size

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

