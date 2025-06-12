from typing import List, Optional
from pydantic import BaseModel

class MessageRequest(BaseModel):
    role: str
    content: str
class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[MessageRequest]
    stream: Optional[bool] = False
    tools: Optional[list] = None
    tool_choice: Optional[str] = None
    json_schema: Optional[dict] = None    
    max_tokens: Optional[int] = None
    stop: Optional[list] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
class ModelDownloadRequest(BaseModel):
    model: str