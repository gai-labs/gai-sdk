import time
import uuid
import threading
from pydantic import BaseModel, model_validator, Field, create_model
from typing import Type, Annotated, Any, Literal, Optional, final
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

# Prevent race conditions when registering bodies
# This is a simple lock to ensure that only one thread can modify the _BODY_CLASSES list
_registry_lock = threading.Lock()


def register_body(cls: Type[BaseModel]) -> Type[BaseModel]:
    with _registry_lock:
        # drop any earlier class with this __name__
        _BODY_CLASSES[:] = [c for c in _BODY_CLASSES if c.__name__ != cls.__name__]
        _BODY_CLASSES.append(cls)
    return cls


# ─── Built-in Message Bodies ─────────────────────────────────────────────────────────


# Default Body -----------------------------------------------------------------------------------


@register_body
class DefaultBodyPydantic(BaseModel, MessageBodyMixin):
    type: Literal["default"] = "default"
    content: Optional[Any] = None


# Monologue Body -----------------------------------------------------------------------------------


@register_body
class MonologueBodyPydantic(BaseModel, MessageBodyMixin):
    type: Literal["monologue"] = "monologue"
    state_name: str
    step_no: int
    content_type: Literal["text", "image", "video", "audio"] = "text"
    role: str
    content: Any


# Chat Send Body -----------------------------------------------------------------------------------


@register_body
class ChatSendBodyPydantic(BaseModel, MessageBodyMixin):
    type: Literal["chat.send"] = "chat.send"
    dialogue_id: Optional[str]
    round_no: Optional[int]
    step_no: Optional[int]
    message_id: Optional[str]
    content_type: Literal["text", "image", "video", "audio"] = "text"
    role: Literal["user", "assistant"] = "user"
    content: Optional[str]


# Chat Reply Body -----------------------------------------------------------------------------------


@register_body
class ChatReplyBodyPydantic(BaseModel, MessageBodyMixin):
    type: Literal["chat.reply"] = "chat.reply"
    dialogue_id: Optional[str]
    round_no: Optional[int]
    step_no: Optional[int]
    message_id: Optional[str]
    chunk_no: Optional[int]
    chunk: Optional[str]
    content_type: Literal["text", "image", "video", "audio"] = "text"
    role: Literal["user", "assistant"] = "assistant"
    content: Optional[str]


# Handshake Body -----------------------------------------------------------------------------------


class OrchStepPydantic(BaseModel):
    step_type: Literal["chat.send", "chat.reply"] = Field(
        ...,
        description="Type of message step, either 'chat.send' for initiating or 'chat.reply' for responding.",
    )
    sender: str = Field(
        ..., description="Name of the node or agent sending the message."
    )
    recipient: str = Field(
        ..., description="Name of the node or agent intended to receive the message."
    )
    step_no: int = Field(
        ...,
        description="Step number in the conversation flow, indicating order of execution.",
    )
    is_pm: bool = Field(
        ...,
        description="Indicates whether this step is a private message (True) or not (False).",
    )


class OrchPlanPydantic(BaseModel):
    dialogue_id: str = Field(
        ..., description="Unique identifier for the dialogue session."
    )
    round_no: int = Field(
        ...,
        description="The round number within the current dialogue context. A round begins with the User and ends with the last agent's reply.",
    )
    curr_step_no: int = Field(
        0, description="Current step number being executed in the dialogue plan."
    )
    steps: list[OrchStepPydantic] = Field(
        default_factory=list,
        description="Ordered list of steps to execute in this plan.",
    )
    participants: list[str] = Field(
        ..., description="List of participant node names involved in the dialogue."
    )
    flow_type: Literal["poll", "chain"] = Field(
        ...,
        description="Type of dialogue flow: 'poll' for simultaneous turns, 'chain' for sequential turns.",
    )


@register_body
class HandshakeBodyPydantic(BaseModel, MessageBodyMixin):
    type: Literal["system.handshake"] = "system.handshake"
    body: OrchPlanPydantic


@register_body
class HandshakeAckBodyPydantic(BaseModel, MessageBodyMixin):
    type: Literal["system.handshake_ack"] = "system.handshake_ack"
    body: OrchPlanPydantic


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
    # globals()["MessagePydantic"] = model
    return model


# Run this to register the built-in types
MessagePydantic = get_message_cls()
