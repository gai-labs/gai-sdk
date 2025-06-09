import pytest
from unittest.mock import patch, mock_open, MagicMock,call
from gai.lib.config import GaiConfig, config_helper
from gai.lib.tests import make_local_tmp, get_local_datadir

@pytest.fixture
def local_datadir(request):
    """
    Create a tmp/ folder inside the directory of the test file that uses this fixture.
    """
    return get_local_datadir(request)

### GaiConfig should load from "~/.gai/gai.yml" by default

@patch("gai.lib.config.config_helper.get_app_path", return_value="~/.gai")
@patch("builtins.open", new_callable=mock_open, read_data="version: 1.0\ngai_url: http://localhost")
@patch("yaml.load", return_value={"version": "1.0"})
def test_gaiconfig_from_path_default(mock_yaml_load, mock_file, mock_app_path):
    # Load GaiConfig via our new helper
    config = config_helper.get_gai_config()

    # Should have opened exactly ~/.gai/gai.yml
    mock_file.assert_called_once_with("~/.gai/gai.yml", 'r')
    assert len(mock_file.call_args_list) == 1


### GaiConfig should load from custom path
    
@patch("gai.lib.config.config_helper.get_app_path", return_value="~/.gai")
@patch("builtins.open", new_callable=mock_open, read_data="version: 1.0\ngai_url: http://localhost")
@patch("yaml.load", return_value={"version": "1.0"})
def test_gaiconfig_from_path_custom(mock_yaml_load, mock_file, mock_app_path, local_datadir):
    # Load GaiConfig via our new helper, with a custom file_path
    config = config_helper.get_gai_config(f"{local_datadir}/gai.yml")

    # Should have opened exactly the custom path
    mock_file.assert_called_once_with(f"{local_datadir}/gai.yml", 'r')
    assert len(mock_file.call_args_list) == 1

# Client Config ――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――    

mock_client_data= {
    "version": "1.0",
    "gai_url": "http://localhost:8080",
    "logging": {
        "level": "DEBUG",
        "format": "%(levelname)s - %(message)s"
    },
    "clients": {
        "ttt": {
            "ref": "llama3.1",
        },
        "llama3.1": {
            "type": "llama3.1",
            "engine": "ollama",
            "model": "llama3.1",
            "name": "llama3.1",
            "client_type": "ollama"
        }
    }
}

# ── 1) Default path lookup ────────────────────────────────────────────────────
@patch("gai.lib.config.config_helper.get_app_path", return_value="~/.gai")
@patch("builtins.open",   new_callable=mock_open, read_data="version: 1.0\ngai_url: http://localhost")
@patch("yaml.load",       return_value=mock_client_data)
def test_clientconfig_from_path_default(mock_yaml_load, mock_file, mock_app_path):
    # should read ~/.gai/gai.yml and resolve the "ttt" alias → llama3.1
    cfg = config_helper.get_client_config("ttt")
    mock_file.assert_called_once_with("~/.gai/gai.yml", "r")
    assert cfg.name == "llama3.1"


# ── 2) Custom path lookup ─────────────────────────────────────────────────────
@patch("gai.lib.config.config_helper.get_app_path", return_value="~/.gai")
@patch("builtins.open",   new_callable=mock_open, read_data="version: 1.0\ngai_url: http://localhost")
@patch("yaml.load",       return_value=mock_client_data)
def test_clientconfig_from_custom_path(mock_yaml_load, mock_file, mock_app_path):
    custom = "/tmp/gai.yml"
    cfg = config_helper.get_client_config("ttt", file_path=custom)
    mock_file.assert_called_once_with(custom, "r")
    assert cfg.name == "llama3.1"


# ── 3) Dict‐only (no file read) ────────────────────────────────────────────────
@patch("builtins.open", new_callable=mock_open)
def test_clientconfig_from_dict(mock_file):
    raw = {
        "type": "ttt",
        "engine": "ollama",
        "model": "llama3.1",
        "name": "llama3.1",
        "client_type": "ollama"
    }
    cfg = config_helper.get_client_config(raw)
    # must not open any file
    assert len(mock_file.call_args_list) == 0
    assert cfg.name == "llama3.1"


# ── 4) Alias resolution ───────────────────────────────────────────────────────
@patch("gai.lib.config.config_helper.get_app_path", return_value="~/.gai")
@patch("builtins.open",   new_callable=mock_open, read_data="version: 1.0\ngai_url: http://localhost")
@patch("yaml.load",       return_value=mock_client_data)
def test_clientconfig_resolve_ref(mock_yaml_load, mock_file, mock_app_path):
    # direct lookup
    direct = config_helper.get_client_config("llama3.1")
    assert direct.name == "llama3.1"
    # alias lookup
    alias = config_helper.get_client_config("ttt")
    assert alias.name == "llama3.1"

# Generator Config ――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――    

mock_generator_data= {
    "version": "1.0",
    "generators": {
        "ttt": {
            "ref": "ttt-exllamav2-dolphin",
        },
        "ttt-exllamav2-dolphin": {
            "type": "ttt",
            "engine": "exllamav2",
            "model": "dolphin",
            "name": "ttt-exllamav2-dolphin",
            "module":{
                "name": "gai.ttt.server.gai_exllamav2",
                "class": "GaiExllamav2"
            }
        }
    }
}

# ── 1) Default path lookup ────────────────────────────────────────────────────
@patch("gai.lib.config.config_helper.get_app_path", return_value="~/.gai")
@patch("builtins.open",   new_callable=mock_open, read_data="version: 1.0\ngai_url: http://localhost")
@patch("yaml.load",       return_value=mock_generator_data)
def test_generatorconfig_from_path_default(mock_yaml_load, mock_file, mock_app_path):
    cfg = config_helper.get_generator_config("ttt-exllamav2-dolphin")
    mock_file.assert_called_once_with("~/.gai/gai.yml", "r")
    assert cfg.module.name == "gai.ttt.server.gai_exllamav2"


# ── 2) Custom path lookup ─────────────────────────────────────────────────────
@patch("gai.lib.config.config_helper.get_app_path", return_value="~/.gai")
@patch("builtins.open",   new_callable=mock_open, read_data="version: 1.0\ngai_url: http://localhost")
@patch("yaml.load",       return_value=mock_generator_data)
def test_generatorconfig_from_custom_path(mock_yaml_load, mock_file, mock_app_path):
    custom_path = "/tmp/gai.yml"
    cfg = config_helper.get_generator_config("ttt-exllamav2-dolphin", file_path=custom_path)
    mock_file.assert_called_once_with(custom_path, "r")
    assert cfg.module.name == "gai.ttt.server.gai_exllamav2"


# ── 3) Dict-only (no file read) ────────────────────────────────────────────────
@patch("builtins.open", new_callable=mock_open)
def test_generatorconfig_from_dict(mock_file):
    raw = {
        "type": "ttt",
        "engine": "llamacpp",
        "model": "dolphin",
        "name": "ttt-llamacpp-dolphin",
        "module": {
            "name": "gai.ttt.server.gai_llamacpp",
            "class": "GaiLlamaCpp"
        }
    }
    cfg = config_helper.get_generator_config(name_or_config=raw)
    # no file should be opened
    assert len(mock_file.call_args_list) == 0
    assert cfg.module.name == "gai.ttt.server.gai_llamacpp"


# ── 4) Alias resolution ───────────────────────────────────────────────────────
@patch("gai.lib.config.config_helper.get_app_path", return_value="~/.gai")
@patch("builtins.open",   new_callable=mock_open, read_data="version: 1.0\ngai_url: http://localhost")
@patch("yaml.load",       return_value=mock_generator_data)
def test_generatorconfig_resolve_ref(mock_yaml_load, mock_file, mock_app_path):
    # direct lookup
    direct = config_helper.get_generator_config("ttt-exllamav2-dolphin")
    assert direct.name == "ttt-exllamav2-dolphin"
    # alias lookup should resolve to the same
    alias = config_helper.get_generator_config("ttt")
    assert alias.name == "ttt-exllamav2-dolphin"
    
