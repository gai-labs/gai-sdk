import time
import uuid
from pydantic import BaseModel, model_validator, Field
from typing import Annotated, Any, Literal, Optional, Union, final, TypeAlias
from gai.lib.constants import DEFAULT_GUID
from .message_counter import MessageCounter
from gai.lib.logging import getLogger
logger = getLogger(__name__)

# Header Class -----------------------------------------------------------------------------------

@final
class MessageHeaderPydantic(BaseModel):
    """
    This is the envelope header for a message.
    Unlike normal LLM messages, GAI messages are directed and have a sender and recipient.    
    `sender` and `recipient` refers to the `name` not `role` of the participants.
    If they are not specified, they default to "User" and "Assistant" respectively (with Capitalization).
    """
    sender: str = "User"
    recipient: str = "Assistant"
    timestamp: Optional[float]=Field(default_factory=time.time)
    order: Optional[int] = 0  # used to order messages in a dialogue to prevent missing or duplicate messages

# Mixin ------------------------------------------------------------------------------------

class MessageBodyMixin:
    @model_validator(mode="before")
    @classmethod
    def set_message_fields(cls, values):
        """
        This method sets the message_no and message_id fields based on the dialogue_id.
        It uses the MessageCounter to get the next message number.
        """
        if isinstance(values, dict):
            dialogue_id = values.get("dialogue_id", DEFAULT_GUID)
            mc = MessageCounter()
            message_no = mc.get()
            values["message_no"] = message_no
            values["message_id"] = f"{dialogue_id}.{message_no}"
        return values

# Default Class -----------------------------------------------------------------------------------        

class DefaultBodyPydantic(BaseModel,MessageBodyMixin):
    type: Literal["default"] = "default"
    content: Optional[Any]

# State Class -----------------------------------------------------------------------------------        

class StateBodyPydantic(BaseModel, MessageBodyMixin):
    type: Literal["state"] = "state"
    state_name: str
    step_no: int
    content_type: Literal["text", "image", "video", "audio"]="text"
    role: str
    content: Any

# Send Class -----------------------------------------------------------------------------------        

class SendBodyPydantic(BaseModel, MessageBodyMixin):
    type: Literal["send"] = "send"
    dialogue_id: Optional[str] = DEFAULT_GUID
    message_no: Optional[int] = None    # Will be set by validator
    message_id: Optional[str] = None    # Will be set by validator
    content_type: Literal["text", "image", "video", "audio"]="text"
    content: str

# Reply Class -----------------------------------------------------------------------------------        

class ReplyBodyPydantic(BaseModel, MessageBodyMixin):
    type: Literal["reply"] = "reply"
    dialogue_id: Optional[str] = DEFAULT_GUID
    message_no: Optional[int] = None  # Will be set by validator
    message_id: Optional[str] = None  # Will be set by validator
    chunk_no: Optional[int] = 0
    chunk: Optional[str] = "<eom>"
    content_type: Literal["text", "image", "video", "audio"]="text"
    content: Optional[str] = None
 
# Message Class -----------------------------------------------------------------------------------
UnionBodyType:TypeAlias = Annotated[Union[DefaultBodyPydantic, StateBodyPydantic, SendBodyPydantic, ReplyBodyPydantic],Field(discriminator="type")]

class MessagePydantic(BaseModel):
    """
    Default message class for all GAI messages.
    The only specific part of the message is the body(payload) which is a discriminated union of different message types by "message_type".
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    header: MessageHeaderPydantic = Field(default_factory=MessageHeaderPydantic)  
    body: UnionBodyType
