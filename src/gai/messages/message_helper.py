import re
from typing import Any, Literal, Type, TypeVar

from gai.lib.constants import DEFAULT_GUID
from pydantic import BaseModel
from .typing import (
    DefaultBodyPydantic,
    MessageHeaderPydantic,
    # ReplyBodyPydantic,
    # SendBodyPydantic,
    MessagePydantic
)


def create_message(
    role: Literal["user", "assistant", "system"], content: str
) -> MessagePydantic:
    """
    Create a message in {"role":"...","content":"..."} format.
    Use this for standard chat messages.

    Args:
        role (str): The role of the sender in lowercase (e.g., "user", "assistant","system").
        content (str): The content of the message.
    Returns:
        MessagePydantic: A message object with the specified role and content.
    """
    if not role:
        raise ValueError("Role must be specified")
    if not content:
        raise ValueError("Content must be specified")
    name = role.capitalize()  # Capitalize the role for the header
    recipient = "Assistant" if name == "User" else "User"
    return MessagePydantic(
        header=MessageHeaderPydantic(sender=name, recipient=recipient),
        body=DefaultBodyPydantic(content=content),
    )


def convert_to_chat_messages(messages: list[MessagePydantic]) -> list[dict[str, Any]]:
    """
    Convert a list of messages to chat messages.

    Args:
        messages (list[MessagePydantic]): A list of messages to convert.

    Returns:
        list[MessagePydantic]: A list of chat messages.
    """
    if not messages:
        return []
    chat_messages = []
    for m in messages:
        if hasattr(m.body, "content"):
            # Only if message has content
            if m.body.role == "system":
                # clean up whitespace from system messages
                m.body.content = re.sub(r"\s+", " ", m.body.content)
            chat_messages.append(
                {"role": m.body.role, "content": m.body.content})

    return chat_messages


MessagePydanticT = TypeVar("MessagePydanticT", bound=BaseModel)


def json(list: list[MessagePydanticT]) -> str:
    """
    Convert a list of messages to JSON format.

    Args:
        list (list[MessagePydantic]): A list of messages to convert.

    Returns:
        str: A JSON string representation of the messages.
    """
    import json

    return json.dumps([message.model_dump() for message in list], indent=4)


def unjson(json_str: str, MessagePydantic_cls: Type[MessagePydanticT]) -> list[MessagePydanticT]:
    """
    Convert a JSON string to a list of messages.

    Args:
        json_str (str): A JSON string representation of messages.
        MessagePydantic_cls (Type[MessagePydanticT]): The dynamic MessagePydantic class to validate against.
    Returns:
        list[MessagePydanticT]: A list of messages of dynamic MessagePydantic class type.
    """
    import json as json_lib

    return [
        MessagePydantic_cls.model_validate(message) for message in json_lib.loads(json_str)
    ]


def extract_recap(
    messages: list[MessagePydantic], last_n: int, max_recap_size: int
) -> str:
    """
    Extract a recap of the last N messages, constrained by max_recap_size.
    Instead of showing the sender role, it uses sender name.
    This is to facilitate multi-agent dialogues so that agent can tell apart who said what.
    For example, 

    User: <content>
    Sara: <content>

    Instead of

    [
    {"role": "user", "content": "<content>"},
    {"role": "assistant", "content": "<content>"}
    ]

    Args:
        messages (list[MessagePydantic]): The full message history.
        last_n (int): Number of most recent messages to consider.
        max_recap_size (int): Maximum character length for the recap.

    Returns:
        str: A recap string containing recent messages up to the size limit.
    """
    # Step 1: Get the last N messages
    recent_messages = messages[-last_n:]

    # Step 2: Convert messages to dialogue format
    recap_lines = []
    total_len = 0
    for m in recent_messages:
        if hasattr(m.body, "content") and isinstance(m.body.content, str):
            line = f"{m.header.sender}: {m.body.content.strip()}"
            if total_len + len(line) > max_recap_size:
                break
            recap_lines.append(line)
            total_len += len(line)

    # Step 3: Join the lines into a single string
    return "\n".join(recap_lines)
