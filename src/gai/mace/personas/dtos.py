import uuid
from pydantic import BaseModel, ConfigDict, Field, Json
from typing import List, Literal, Optional, Union, Any
from gai.lib.config import GaiClientConfig
    
class PersonaClassPydantic(BaseModel):
    model_config = ConfigDict(from_attributes=True)  # Allows the model to work with ORM objects
    id: str
    class_name: str
    job_description: str
    self_introduction: str
    diagram_name: str
    manifest_name: str
    tool_name: Optional[str]=None
    
    @classmethod
    def from_dict(cls, data: dict) -> 'PersonaClassPydantic':
        if 'id' not in data:
            data['id'] = str(uuid.uuid4())
        return cls(**data)

class ComponentPydantic(BaseModel):
    model_config = ConfigDict(from_attributes=True)  # Allows the model to work with ORM objects
    id: str
    component_type: str
    name: str = Field(..., max_length=50)
    desc: Optional[str] = None
    content: Optional[str|dict] = None
    creator_id: str
    usage_type: str
    read_only: bool = True

#----------------- Prompt DTOs -----------------#

class PromptPydantic(BaseModel):
    prompt: str
    class_filter: list[str] = []

#----------------- Tool DTOs -----------------#

class ToolParameterPydantic(BaseModel):
    type: str
    description: Optional[str]

class ToolParametersPydantic(BaseModel):
    type: Literal["object"]
    properties: dict[str,ToolParameterPydantic]
    required: list[str]=[]

class ToolDefinitionPydantic(BaseModel):
    name: str
    description: str
    parameters: ToolParametersPydantic

class ToolPydantic(BaseModel):
    type: Literal["function"]
    function: ToolDefinitionPydantic

#----------------- Image DTOs -----------------#

class PersonaImagePydantic(BaseModel):
    image_name: str
    image_type: str = "png"
    image_prompt: str
    negative_prompt: str
    image512: bytes
    image256: Optional[bytes]
    image128: Optional[bytes]
    image64: Optional[bytes]

class GenerateImageRequestPydantic(BaseModel):
    persona_name: str=Field(..., max_length=255)
    persona_traits: Optional[list[str]] = []
    persona_sex: Optional[Literal["male","female"]] = None
    image_styles: Optional[list[str]] = []
    
class GenerateImageResponsePydantic(BaseModel):
    image_name: str
    image_type: str = "png"
    image_prompt: str
    negative_prompt: str
    data_url_512: str
    data_url_256: str
    data_url_128: str
    data_url_64: str
    
class DataUrlPydantic(BaseModel):
    data_url: str
    
#----------------- Agent DTOs -----------------#
    
class PersonaPydantic(BaseModel):
    model_config = ConfigDict(from_attributes=True)  # Allows the model to work with ORM objects
    id: str
    owner_id: str
    name: str
    job_description: str
    self_introduction: str
    sex: Optional[Literal["male","female"]]=None
    skills: Optional[list[str]]=None
    traits: Optional[list[str]]=None
    persona_class: str
    llm_config: GaiClientConfig
    state_diagram: str
    states_manifest: dict
    tools_dict: Optional[dict]=None
    portraits: Optional[PersonaImagePydantic]=None

#----------------- Documents DTOs -----------------#

class FlattenedAgentDocumentPydantic(BaseModel):
    model_config = ConfigDict(from_attributes=True)  # Allows the model to work with ORM objects
    Id: Optional[str] = None
    PersonaId: Optional[str]
    FileName: Optional[str]
    FileType: Optional[str] = None
    Source: Optional[str] = None
    ByteSize: Optional[int] = None
    Title: Optional[str] = None
    Abstract: Optional[str] = None
    Authors: Optional[str] = None
    Publisher: Optional[str] = None
    PublishedDate: Optional[str] = None
    Comments: Optional[str] = None
    Keywords: Optional[str] = None
    ChunkGroupId: Optional[str] = None
    ChunkSize: Optional[int] = None
    ChunkOverlap: Optional[int] = None
    ChunkCount: Optional[int] = None

class ProvisionPersonaRequestPydantic(BaseModel):
    name: str
    llm: Optional[str]=None
    agent_class: str
    traits: Optional[list[str]]=None
    sex: Optional[Literal["male","female"]]=None
    job_description: str
    self_introduction: Optional[str]=None
    tool: Optional[str]=None
    state_diagram: Optional[str]=None
    states_manifest: Optional[str]=None
    image_styles: Optional[list[str]]=None
    