import subprocess
import os
from io import BytesIO


def play_audio(response):
    ffplay_cmd = ["ffplay", "-nodisp", "-autoexit", "-"]
    ffplay_proc = subprocess.Popen(
        ffplay_cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

    # Check if response is bytes
    if isinstance(response, bytes):
        # Convert bytes to BytesIO
        response = BytesIO(response)
        for chunk in iter(lambda: response.read(1024), b''):
            if chunk:
                ffplay_proc.stdin.write(chunk)
    else:
        for chunk in response.raw.stream(1024, decode_content=False):
            if chunk:
                ffplay_proc.stdin.write(chunk)

    # close on finish
    ffplay_proc.stdin.close()
    ffplay_proc.wait()


def save_audio(response, file_path):
    # Check if response is bytes
    if isinstance(response, bytes):
        with open(file_path, 'wb') as out_file:
            out_file.write(response)
    else:
        with open(file_path, 'wb') as out_file:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:  # filter out keep-alive new chunks
                    out_file.write(chunk)

    print(f"Audio saved at {os.path.abspath(file_path)}")
