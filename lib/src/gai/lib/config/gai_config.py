import yaml
from pydantic import BaseModel
from typing import Dict, Optional
from gai.lib.logging import getLogger
from .gai_client_config import GaiClientConfig
from .gai_generator_config import GaiGeneratorConfig
from .gai_tool_config import GaiToolConfig
logger = getLogger(__name__)

class LogConfig(BaseModel):
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    datefmt: str = "%Y-%m-%d %H:%M:%S"
    filename: Optional[str] = None
    filemode: str = "a"
    stream: str = "stdout"
    loggers: Optional[Dict] = None

class GaiConfig(BaseModel):
    version: str
    gai_url: Optional[str] = None
    logging: Optional[LogConfig] = None
    clients: Optional[dict[str,GaiClientConfig] ] = {}
    generators: Optional[dict[str,GaiGeneratorConfig] ] = {}
    tools: Optional[dict[str,GaiToolConfig] ] = {}
    class Config:
        extra = "ignore"
    
    def to_yaml(self):
        
        # Convert class_ to class before saving
        
        jsoned = self.model_dump()
        if jsoned.get("generators",None):

            # Ensure full module and source fields are preserved

            for g in jsoned["generators"].values():
                if isinstance(g.get("module"), BaseModel):
                    g["module"] = g["module"].model_dump()

                source = g.get("source")
                if isinstance(source, BaseModel):
                    g["source"] = source.model_dump()

                # Convert class_ to class for YAML output
                if g.get("module") and g["module"].get("class_"):
                    g["module"]["class"] = g["module"].pop("class_")        
        
        y=yaml.dump(jsoned, sort_keys=False,indent=4)
        return y
