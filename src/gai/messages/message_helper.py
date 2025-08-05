import re
from typing import Any, Literal, Type, TypeVar, Optional
from gai.lib.logging import getLogger
from gai.lib.constants import DEFAULT_GUID
from pydantic import BaseModel

from .typing import (
    DefaultBodyPydantic,
    MessageHeaderPydantic,
    # ReplyBodyPydantic,
    # SendBodyPydantic,
    MessagePydantic,
    MonologueBodyPydantic,
)

logger = getLogger(__name__)


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
        # If message has content and role, convert to chat message format
        if isinstance(m.body, dict) and m.body.get("content") and m.body.get("role"):
            role = m.body["role"]
            content = m.body["content"]
            if role == "system":
                # clean up whitespace from system messages
                content = re.sub(r"\s+", " ", content)
            chat_messages.append({"role": role, "content": content})

        elif hasattr(m.body, "content") and hasattr(m.body, "role"):
            role = m.body.role
            content = m.body.content
            if role == "system":
                # clean up whitespace from system messages
                content = re.sub(r"\s+", " ", content)
            chat_messages.append({"role": role, "content": content})
        else:
            # This could be a control message eg. type="system.rollcall"
            pass

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


def unjson(
    json_str: str, MessagePydantic_cls: Type[MessagePydanticT]
) -> list[MessagePydanticT]:
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
        MessagePydantic_cls.model_validate(message)
        for message in json_lib.loads(json_str)
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


def get_messages_length(chat_messages: list[dict] = None) -> int:
    """
    Returns the length of the messages's content in characters.
    """
    import json

    if not chat_messages:
        return 0
    if not chat_messages[0].get("role") or not chat_messages[0].get("content"):
        raise ValueError(
            "Invalid chat message format. Each message must have 'role' and 'content'."
        )
    return len(json.dumps(chat_messages))


def fix_messages(chat_messages: list[dict]) -> list[dict]:
    """
    Fix the messages such that out of order tool_use and tool_result messages are removed.
    """

    if not chat_messages:
        return []

    if chat_messages[0].get("role") != "user":
        raise ValueError(
            "message_helper.fix_messages: First message is not a user message."
        )

    # Remove any tool_use messages that are not followed by a tool_result message
    fixed_messages = []
    expected_role = "user"  # Start with user role
    for i, msg in enumerate(chat_messages):
        if msg.get("role") != expected_role:
            logger.warning(
                f"message_helper.fix_messages: Skipping duplicated role {msg.get('role')} message at index {i}"
            )
            continue

        if expected_role == "user":
            expected_role = "assistant"
        else:
            expected_role = "user"

        if msg.get("role") == "assistant" and isinstance(msg.get("content"), list):
            # Check if this is a tool_use message
            has_tool_use = any(
                item.get("type") == "tool_use" for item in msg["content"]
            )
            if has_tool_use:
                # Check if the next message is a user tool_result message
                if (
                    i + 1 < len(chat_messages)
                    and chat_messages[i + 1].get("role") == "user"
                ):
                    next_content = chat_messages[i + 1].get("content")
                    if isinstance(next_content, list) and any(
                        item.get("type") == "tool_result" for item in next_content
                    ):
                        fixed_messages.append(msg)
            else:
                # If it's not a tool_use message, just append it
                fixed_messages.append(msg)
        else:
            fixed_messages.append(msg)

    if fixed_messages[-1].get("role") == "assistant":
        # If the last message is an assistant message, remove it
        fixed_messages.pop()

    return fixed_messages


def shrink_messages(chat_messages: list[dict], limit_len: int = 300000) -> list[dict]:
    """
    Shrink the messages to fit within the character limit but preserve the earliest user message.
    This is useful for long conversations where we want to keep the context but not exceed limits.

    Args:
        chat_messages (list[dict]): A list of chat messages.
        limit_len (int): The maximum length of the messages in characters.

    Returns:
        list[dict]: A list of shrunk chat messages.
    """

    # If no messages are provided, return an empty list

    if not chat_messages:
        return []

    # Step 1: Find the earliest user message

    earliest_user_message = chat_messages[0]
    if not earliest_user_message.get("role") == "user":
        raise ValueError(
            "message_helper.shrink_messages: First message is not a user message."
        )

    # If contains only one message, return it as is
    chat_messages_copy = chat_messages.copy()
    if len(chat_messages_copy) == 1:
        return chat_messages_copy

    # Step 2: Pop the first user message and shrink the rest starting from the end

    earliest_user_message = chat_messages_copy.pop(0)  # Remove the first user message
    init_length = get_messages_length([earliest_user_message])

    # Reverse the list to start popping from the end

    chat_messages_copy.reverse()
    total_len = get_messages_length(chat_messages_copy)
    while total_len + init_length > limit_len:
        # Remove assistant message
        if not chat_messages_copy:
            # Ran out of messages
            break
        while chat_messages_copy and chat_messages_copy[-1].get("role") != "assistant":
            # This is to deal with corrupted messages where user messages are present instead of assistant message.
            chat_messages_copy.pop()
        if chat_messages_copy and chat_messages_copy[-1].get("role") == "assistant":
            chat_messages_copy.pop()

        # Remove user message
        if not chat_messages_copy:
            # Ran out of messages
            break
        while chat_messages_copy and chat_messages_copy[-1].get("role") != "user":
            # This is to deal with corrupted messages where assistant messages are present instead of user message.
            chat_messages_copy.pop()
        if chat_messages_copy and chat_messages_copy[-1].get("role") == "user":
            chat_messages_copy.pop()

        # Update the total length after popping messages
        total_len = get_messages_length(chat_messages_copy)

    # Push the earliest user message back to the front
    chat_messages_copy.reverse()  # Reverse back to original order
    chat_messages_copy.insert(0, earliest_user_message)

    chat_messages_copy = fix_messages(chat_messages_copy)

    # Step 3: Validate the messages after shrinking
    # This is highly error-prone, so we validate the messages immediately after shrinking.
    validate_tool_messages(chat_messages_copy)

    return chat_messages_copy


def validate_tool_messages(chat_messages: list[dict]) -> bool:
    # a) First message is user message
    if not chat_messages:
        raise ValueError(
            "message_helper.validate_tool_messages: chat_messages should not be empty and should have at least 1 user message."
        )

    if chat_messages and chat_messages[0].get("role") != "user":
        raise ValueError(
            "message_helper.validate_tool_messages: First message is not a user message."
        )

    expected_role = "user"
    for i, msg in enumerate(chat_messages):
        # b) Subsequent messages should alternate between user and assistant

        if msg.get("role") != expected_role:
            raise ValueError(
                f"message_helper.validate_tool_messages: Expected role '{expected_role}' but found '{msg.get('role')}' at index {i}."
            )

        if expected_role == "user":
            expected_role = "assistant"
        else:
            expected_role = "user"

        # c) If current message is an assistant tool_use message then the next message should be a user tool_result message with matching tool_use_id

        if msg.get("role") == "assistant" and isinstance(msg.get("content"), list):
            for assistant_item in msg["content"]:
                if assistant_item.get("type") == "tool_use":
                    # Unexpected end of messages

                    if len(chat_messages) <= i + 1:
                        raise ValueError(
                            "message_helper.validate_tool_messages: Expecting a user tool_result message but messages ended abruptly."
                        )

                    # Unexpected role after "assistant"

                    if chat_messages[i + 1]["role"] != "user":
                        raise ValueError(
                            f"message_helper.validate_tool_messages: Expecting a user tool_result message but found a {chat_messages[i + 1]['role']} message at index {i + 1}."
                        )

                    next_content = chat_messages[i + 1].get("content")

                    # Next message content should be a list

                    if not isinstance(next_content, list):
                        raise ValueError(
                            f"message_helper.validate_tool_messages: Expecting a list of content in user message at index {i + 1} but found {next_content}."
                        )

                    # Next message should contain tool_result with matching tool_use_id

                    for next_item in next_content:
                        if next_item.get("type") == "tool_result":
                            if next_item.get("tool_use_id") != assistant_item.get("id"):
                                raise ValueError(
                                    f"message_helper.validate_tool_messages: Expecting user `tool_result` message to contain tool_use_id=`{next_item.get('tool_use_id')}` does not match tool_use.id {assistant_item.get('id')} at index {i + 1}."
                                )
                        else:
                            raise ValueError(
                                f"message_helper.validate_tool_messages: Expecting user message `tool_result` type after assistant `tool_use` message but found user message `{next_item.get('type')}` type at index {i + 1}."
                            )

        # d) If current message is a user tool_result message then the previous message should be an assistant tool_use message with matching id

        if msg.get("role") == "user" and isinstance(msg.get("content"), list):
            for user_item in msg["content"]:
                if user_item.get("type") == "tool_result":
                    # Unexpected start of messages

                    if i == 0:
                        raise ValueError(
                            "message_helper.validate_tool_messages: Expecting an assistant tool_use message before user tool_result message but messages started with a user message."
                        )

                    # Previous message should be an assistant tool_use message

                    previous_message = chat_messages[i - 1]
                    if previous_message.get("role") != "assistant":
                        raise ValueError(
                            f"message_helper.validate_tool_messages: Expecting an assistant tool_use message before user tool_result message but found a {previous_message['role']} message at index {i - 1}."
                        )

                    previous_content = previous_message.get("content")

                    # Previous message content should be a list

                    if not isinstance(previous_content, list):
                        raise ValueError(
                            f"message_helper.validate_tool_messages: Expecting a list of content in assistant message at index {i - 1} but found {previous_content}."
                        )

                    # Previous message should contain tool_use with matching id
                    found_tool_use = False
                    for previous_item in previous_content:
                        if previous_item.get("type") == "tool_use":
                            found_tool_use = True
                            if previous_item.get("id") != user_item.get("tool_use_id"):
                                raise ValueError(
                                    f"message_helper.validate_tool_messages: Expecting assistant `tool_use` id=`{previous_item.get('id')}` to match user `tool_result` tool_use_id={user_item.get('tool_use_id')} at index {i - 1}."
                                )
                    if not found_tool_use:
                        raise ValueError(
                            f"message_helper.validate_tool_messages: Expecting assistant message `tool_use` type before user `tool_result` message but not found at index {i - 1}."
                        )

    # Finally, last message should be a user message
    if chat_messages and chat_messages[-1].get("role") != "user":
        raise ValueError(
            f"message_helper.validate_tool_messages: Last message is not a user message but a {chat_messages[-1].get('role')} message."
        )

    # # Find all tool_use message ids
    # tool_use_messages = [msg for msg in chat_messages if msg["role"]
    #                      == "assistant" and isinstance(msg["content"], list)]
    # tool_use_message_ids = []
    # for msg in tool_use_messages:
    #     for item in msg["content"]:
    #         if item["type"] == "tool_use":
    #             tool_use_message_ids.append(item["id"])

    # # Find all tool_result message ids
    # tool_result_messages = [msg for msg in chat_messages if msg["role"]
    #                         == "user" and isinstance(msg["content"], list)]
    # tool_result_message_ids = []
    # for msg in tool_result_messages:
    #     for item in msg["content"]:
    #         if item["type"] == "tool_result":
    #             tool_result_message_ids.append(item["tool_use_id"])

    # Find all tool_use_message_ids that do not have a corresponding tool_result_message_id and save them in unmatched_tool_use_message_ids
    # Find all tool_result_message_ids that do not have a corresponding tool_use_message_id and save them in unmatched_tool_result_message_ids
    # print both

    # unmatched_tool_use_message_ids = [
    #     msg_id for msg_id in tool_use_message_ids if msg_id not in tool_result_message_ids]
    # unmatched_tool_result_message_ids = [
    #     msg_id for msg_id in tool_result_message_ids if msg_id not in tool_use_message_ids]
    # if len(unmatched_tool_use_message_ids) > 0:
    #     logger.error(
    #         f"`tool_use` ids were found without `tool_result` blocks immediately after: {', '.join(unmatched_tool_use_message_ids)}.")
    #     return False
    # if len(unmatched_tool_result_message_ids) > 0:
    #     logger.error(
    #         f"unexpected `tool_use_id` found in `tool_result` blocks: {', '.join(unmatched_tool_result_message_ids)}. Each `tool_result` block must have a corresponding `tool_use` block in the previous message.")
    #     return False
    # return True


def create_chat_send_message(
    dialogue_id: str,
    round_no: int,
    step_no: int,
    recipient: str,
    content: str,
    sender: str = "User",
):
    # 3) Create the user message
    named_content = f"{recipient}, {content}"
    user_message = MessagePydantic(
        **{
            "header": {
                "sender": sender,
                "recipient": recipient,
            },
            "body": {
                "type": "chat.send",
                "dialogue_id": dialogue_id,
                "round_no": round_no,
                "step_no": step_no,
                "role": "user",
                "content": named_content,
            },
        }
    )
    return user_message


def create_chat_reply_message(
    dialogue_id: str,
    round_no: int,
    step_no: int,
    sender: str,
    chunk_no: int,
    chunk: str,
    content: Optional[str] = None,
    recipient: str = "User",
) -> MessagePydantic:
    # If content is provided, then chunk must be "<eom>"
    if chunk != "<eom>" and content is not None:
        raise ValueError("If content is provided, chunk must be '<eom>'.")

    # If chunk is "<eom>", then content must not be None
    if chunk == "<eom>" and content is None:
        raise ValueError("If chunk is '<eom>', content can be '' but must not be None.")

    # 5) Create the assistant message
    assistant_message = MessagePydantic(
        **{
            "header": {
                "sender": sender,
                "recipient": recipient,
            },
            "body": {
                "type": "chat.reply",
                "dialogue_id": dialogue_id,
                "round_no": round_no,
                "step_no": step_no,
                "role": "assistant",
                "chunk": chunk,
                "chunk_no": chunk_no,
                "content": content,
            },
        }
    )
    return assistant_message


def create_monologue_user_message(
    recipient: str, state_name: str, state_step: int, content: str
):
    user_message = MessagePydantic(
        **{
            "header": MessageHeaderPydantic(sender="User", recipient=recipient),
            "body": MonologueBodyPydantic(
                state_name=state_name,
                step_no=state_step,
                role="user",
                content=content,
            ),
        }
    )
    return user_message


def create_monologue_assistant_message(
    sender: str, state_name: str, state_step: int, content: str
):
    assistant_message = MessagePydantic(
        **{
            "header": MessageHeaderPydantic(sender=sender, recipient="User"),
            "body": MonologueBodyPydantic(
                state_name=state_name,
                step_no=state_step,
                role="assistant",
                content=content,
            ),
        }
    )
    return assistant_message


def print_chunk(chunk: Any) -> None:
    if chunk:
        if isinstance(chunk, str):
            print(chunk, end="", flush=True)
        else:
            if isinstance(chunk, list):
                for item in chunk:
                    if item.get("name"):
                        print(f'Tool: "{item["name"]}"')
                    if item.get("input"):
                        inputs = item.get("input")
                        if isinstance(inputs, dict):
                            for key, value in inputs.items():
                                if isinstance(value, str):
                                    if len(value) > 100:
                                        print(
                                            f"\tInput: {key} = {value[:100]}... (truncated)"
                                        )
                                    else:
                                        print(f"\tInput: {key} = {value}")
