import time
import uuid
from pydantic import BaseModel, model_validator, Field, create_model
from typing import Type,Annotated, Any, Literal, Optional, Union, final, TypeAlias
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
    timestamp: Optional[float] = Field(default_factory=time.time)
    order: Optional[int] = (
        0  # used to order messages in a dialogue to prevent missing or duplicate messages
    )


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
            if "message_no" not in values:
                values["message_no"] = message_no
                values["message_id"] = f"{dialogue_id}.{message_no}"
        return values

# ─── Registry ────────────────────────────────────────────────────────────────

_BODY_CLASSES: list[Type[BaseModel]] = []

def register_body(cls: Type[BaseModel]) -> Type[BaseModel]:
    # drop any earlier class with this __name__
    _BODY_CLASSES[:] = [c for c in _BODY_CLASSES if c.__name__ != cls.__name__]
    _BODY_CLASSES.append(cls)
    return cls


# ─── Built-in Message Bodies ─────────────────────────────────────────────────────────


# Default Body -----------------------------------------------------------------------------------

@register_body
class DefaultBodyPydantic(BaseModel, MessageBodyMixin):
    type: Literal["default"] = "default"
    content: Optional[Any]


# Monologue Body -----------------------------------------------------------------------------------

@register_body
class MonologueBodyPydantic(BaseModel, MessageBodyMixin):
    type: Literal["monologue"] = "monologue"
    state_name: str
    step_no: int
    content_type: Literal["text", "image", "video", "audio"] = "text"
    role: str
    content: Any


# # Send Class -----------------------------------------------------------------------------------

# @register_body
# class SendBodyPydantic(BaseModel, MessageBodyMixin):
#     type: Literal["send"] = "send"
#     dialogue_id: Optional[str] = DEFAULT_GUID
#     message_no: Optional[int] = None  # Will be set by MessageBodyMixin
#     message_id: Optional[str] = None  # Will be set by MessageBodyMixin
#     content_type: Literal["text", "image", "video", "audio"] = "text"
#     content: Any


# # Reply Class -----------------------------------------------------------------------------------

# @register_body
# class ReplyBodyPydantic(BaseModel, MessageBodyMixin):
#     type: Literal["reply"] = "reply"
#     dialogue_id: Optional[str] = DEFAULT_GUID
#     message_no: Optional[int] = None  # Will be set by MessageBodyMixin
#     message_id: Optional[str] = None  # Will be set by MessageBodyMixin
#     chunk_no: Optional[int] = 0
#     chunk: Optional[str] = "<eom>"
#     content_type: Literal["text", "image", "video", "audio"] = "text"
#     content: Optional[Any] = None

# ─── Registry hookup ─────────────────────────────────────────────────────────

def get_message_cls():
    """
    Call *after* you've defined (and decorated) all your bodies.
    This rebuilds MessagePydantic.body to be a discriminated union
    of every registered body class.
    """
    # 1) Dedupe (if you called register_body twice on the same class)
    unique = list(dict.fromkeys(_BODY_CLASSES))
    # 2) Build the union of all body classes
    union = unique[0]
    for cls in unique[1:]:
        union |= cls  # Python 3.10+ union operator
    # 3) Annotate with discriminator
    BodyType = Annotated[union, Field(discriminator="type")]

    # 4) Create the final MessagePydantic *model* all at once
    model = create_model(
        "MessagePydantic",
        id=(str, Field(default_factory=lambda: str(uuid.uuid4()))),
        header=(MessageHeaderPydantic, Field(default_factory=MessageHeaderPydantic)),
        body=(BodyType, ...),
        __base__=BaseModel,
    )

    # 5) Export it
    #globals()["MessagePydantic"] = model
    return model
    
# Run this to register the built-in types

MessagePydantic = get_message_cls()