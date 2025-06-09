def test_get_downoad_config():
    from gai.chat.lib.generators_utils import get_download_config
    from gai.lib.config import HuggingfaceDownloadConfig

    model = get_download_config({
        "type": "huggingface",
        "repo_id": "bartowski/Llama-3.2-3B-Instruct-exl2",
        "local_dir": "Llama-3.2-3B-Instruct-exl2",
        "revision": "c08d657b27cf0450deaddc3e582be20beec3e62d"
    })

    assert type(model) is HuggingfaceDownloadConfig
    assert model.repo_id == "bartowski/Llama-3.2-3B-Instruct-exl2"
    assert model.local_dir == "Llama-3.2-3B-Instruct-exl2"
    assert model.revision == "c08d657b27cf0450deaddc3e582be20beec3e62d"
    
def test_get_download_config_with_alias():
    from gai.chat.lib.generators_utils import get_download_config
    from gai.lib.config import HuggingfaceDownloadConfig
    
    model = get_download_config("ttt")
    
    assert type(model) is HuggingfaceDownloadConfig
    assert model.repo_id == "bartowski/Dolphin3.0-Llama3.1-8B-exl2"
    assert model.local_dir == "Dolphin3.0-Llama3.1-8B-4_25bpw-exl2"
    assert model.revision == "896301e945342d032ef0b3a81b57f0d5a8bac6fe"
    