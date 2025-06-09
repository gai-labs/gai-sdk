import copy
import os
import yaml
from pydantic import BaseModel, Field
from typing import Optional, Union, Literal
from gai.lib.utils import get_app_path
from gai.lib.logging import getLogger
logger = getLogger(__name__)

class ModuleConfig(BaseModel):
    name: str
    class_: str = Field(alias="class")  # Use 'class' as an alias for 'class_'

    class Config:
        allow_population_by_name = True  # Allow access via both 'class' and 'class_'

