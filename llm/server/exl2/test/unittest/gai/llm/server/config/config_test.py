import os, sys
import pytest
sys.path.insert(0,os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from gai.lib.config import config_helper
from gai.lib.tests import get_local_datadir

@pytest.fixture
def generator_config_path(request):
    datadir = get_local_datadir(request)
    file_path = os.path.join(datadir,"generator_gai.yml")
    return file_path

### GaiGeneratorConfig Tests

def test_can_find_builtin_generator_config(generator_config_path):
    generator_config = config_helper.get_generator_config(name="ttt",file_path=generator_config_path)    
    
    assert generator_config is not None
    assert generator_config.type == "ttt"
    assert generator_config.module.class_ == "GaiExLlamav2"
    assert generator_config.module.name == "gai.llm.server.gai_exllamav2"

def test_can_find_ref_config(generator_config_path):
    generator_config = config_helper.get_generator_config(name="dolphin3.0_llama3.1:4.25bpw:exl2", file_path=generator_config_path)
    assert generator_config is not None
    assert generator_config.type == "ttt"
    assert generator_config.name == "dolphin3.0_llama3.1:4.25bpw:exl2"

    generator_config = config_helper.get_generator_config(name="ttt", file_path=generator_config_path) # This should return the same config as above
    assert generator_config is not None
    assert generator_config.type == "ttt"
    assert generator_config.name == "dolphin3.0_llama3.1:4.25bpw:exl2"
    
@pytest.fixture
def init_gen_config(request):
    from gai.lib.tests import get_local_datadir
    data_dir = get_local_datadir(request)
    source_path = os.path.join(data_dir,"gai.yml")
    
    from gai.lib.tests import make_local_tmp
    tmp_dir = make_local_tmp(request)
    dest_path = os.path.join(tmp_dir,"gai.yml")
    
    import shutil
    shutil.copyfile(source_path, dest_path)
    
    return {"source_path": source_path, "dest_path": dest_path,"data_dir": data_dir}


import yaml
def test_can_load_using_class_alias(init_gen_config):
    """
    Prove that configuration can be load the module class using the alias "class" or "class_" as module dictionary key.
    """
    
    # Using GaiConfig with class_ as "class_"
    
    test_data =  {
        "version": "0.1",
        "generators": {
            "ttt": {
                "type": "ttt",
                "engine": "exllamav2",
                "model": "dolphin3.0_llama3.1:4.25bpw",
                "name": "dolphin3.0_llama3.1:4.25bpw:exl2",
                "module": {
                    "name": "gai.llm.server.gai_exllamav2",
                    "class": "GaiExLlamav2"
                },
                "source": {
                    "type": "huggingface",
                    "repo_id": "bartowski/Dolphin3.0-Llama3.1-8B-exl2",
                    "local_dir": "Dolphin3.0-Llama3.1-8B-4_25bpw-exl2",
                    "revision": "896301e945342d032ef0b3a81b57f0d5a8bac6fe"
                }
            }  
        },
    }
    
    gai_config = config_helper.get_gai_config(test_data)
    assert gai_config.generators["ttt"].module.class_ == "GaiExLlamav2"
    gen_config = config_helper.get_generator_config(test_data["generators"]["ttt"])
    assert gen_config.module.class_ == "GaiExLlamav2"
    
    # Using GaiConfig with class_ as "class"
    
    test_data =  {
        "version": "0.1",
        "generators": {
            "ttt": {
                "type": "ttt",
                "engine": "exllamav2",
                "model": "dolphin3.0_llama3.1:4.25bpw",
                "name": "dolphin3.0_llama3.1:4.25bpw:exl2",
                "module": {
                    "name": "gai.llm.server.gai_exllamav2",
                    "class": "GaiExLlamav2"
                },
                "source": {
                    "type": "huggingface",
                    "repo_id": "bartowski/Dolphin3.0-Llama3.1-8B-exl2",
                    "local_dir": "Dolphin3.0-Llama3.1-8B-4_25bpw-exl2",
                    "revision": "896301e945342d032ef0b3a81b57f0d5a8bac6fe"
                }
            }  
        },
    }
    gai_config = config_helper.get_gai_config(test_data)
    assert gai_config.generators["ttt"].module.class_ == "GaiExLlamav2"
    gen_config = config_helper.get_generator_config(test_data["generators"]["ttt"])
    assert gen_config.module.class_ == "GaiExLlamav2"
    
def test_get_download_config(generator_config_path):
    from gai.lib.config.config_helper import get_download_config
    from gai.lib.config.download_config import HuggingfaceDownloadConfig

    model = get_download_config({
        "type": "huggingface",
        "repo_id": "bartowski/Llama-3.2-3B-Instruct-exl2",
        "local_dir": "Llama-3.2-3B-Instruct-exl2",
        "revision": "c08d657b27cf0450deaddc3e582be20beec3e62d"
    }, file_path=generator_config_path)

    assert type(model) is HuggingfaceDownloadConfig
    assert model.repo_id == "bartowski/Llama-3.2-3B-Instruct-exl2"
    assert model.local_dir == "Llama-3.2-3B-Instruct-exl2"
    assert model.revision == "c08d657b27cf0450deaddc3e582be20beec3e62d"
    
def test_get_download_config_with_alias(generator_config_path):
    from gai.lib.config.config_helper import get_download_config
    from gai.lib.config.download_config import HuggingfaceDownloadConfig

    model = get_download_config("ttt", file_path=generator_config_path)

    assert type(model) is HuggingfaceDownloadConfig
    assert model.repo_id == "bartowski/Dolphin3.0-Llama3.1-8B-exl2"
    assert model.local_dir == "Dolphin3.0-Llama3.1-8B-4_25bpw-exl2"
    assert model.revision == "896301e945342d032ef0b3a81b57f0d5a8bac6fe"
    
