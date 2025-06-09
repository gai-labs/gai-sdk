from unittest.mock import patch, mock_open
from gai.lib.utils import get_rc, get_app_path
import os

@patch("builtins.open", new_callable=mock_open, read_data='{"app_dir": "~/.gai"}')
def test_get_rc(mock_file):
    rc=get_rc()
    
    # Ensure only ~/.gairc was opened
    mock_file.assert_called_once_with(os.path.expanduser("~/.gairc"), 'r')
    assert len(mock_file.call_args_list) == 1
    
    assert rc["app_dir"]=="~/.gai"

@patch("gai.lib.utils.get_rc", return_value={"app_dir": "/tmp"})
def test_get_app_path(mock_get_rc):
    
    # Should return the app_dir from the rc file
    assert get_app_path() == "/tmp"