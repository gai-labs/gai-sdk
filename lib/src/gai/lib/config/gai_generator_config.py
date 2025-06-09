from pydantic import BaseModel
from typing import Dict, Optional
from gai.lib.logging import getLogger
from .config_base import ModuleConfig
from .download_config import DownloadConfig
logger = getLogger(__name__)

class MissingGeneratorConfigError(Exception):
    """Custom Exception with a message"""
    def __init__(self, message):
        super().__init__(message)

class GaiGeneratorConfig(BaseModel):
    type: str
    engine: str
    model: str
    name: str
    hyperparameters: Optional[Dict] = {}
    extra: Optional[Dict] = None
    module: ModuleConfig
    source: Optional[DownloadConfig] = None
    class Config:
        extra = "allow"

    @classmethod
    def get_builtin_config_path(cls, this_file) -> str:
        """
        This method is for server subclass to locate the server config file
        """
        from pathlib import Path
        cfg_file = Path(this_file).resolve().parent / "gai.yml"
        file_path = str(cfg_file)
        return file_path
