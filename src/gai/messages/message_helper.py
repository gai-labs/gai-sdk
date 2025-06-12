from typing import Any, Literal

from gai.lib.constants import DEFAULT_GUID
from .typing import DefaultBodyPydantic, MessageHeaderPydantic, MessagePydantic, ReplyBodyPydantic, SendBodyPydantic

def create_message(role:Literal["user","assistant","system"],content:str) -> MessagePydantic:
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
       header=MessageHeaderPydantic(sender=name,recipient=recipient),
       body=DefaultBodyPydantic(content=content)
    )

def convert_to_chat_messages(messages: list[MessagePydantic]) -> list[dict[str,Any]]:
    """
    Convert a list of messages to chat messages.
    
    Args:
        messages (list[MessagePydantic]): A list of messages to convert.
        
    Returns:
        list[MessagePydantic]: A list of chat messages.
    """
    chat_messages=[]
    for message in messages:
        if message.header.sender == "User":
            role = "user"
        elif message.header.sender == "System":
            role = "system"
        else:
            role = "assistant"
        chat_messages.append({
            "role": role,
            "content": message.body.content
        })
    return chat_messages
        

def create_user_send_message(content:str, recipient:str="Assistant", dialogue_id:str=DEFAULT_GUID) -> MessagePydantic:
    """
    Create a user send message.
    
    Args:
        content (str): The content of the message.
        recipient (str): The recipient of the message, defaults to empty string.
        dialogue_id (str): The dialogue ID, defaults to "default-guid".
        
    Returns:
        MessagePydantic: A message object with type "send".
    """
    return MessagePydantic(
       header=MessageHeaderPydantic(sender="User",recipient=recipient),
       body=SendBodyPydantic(
           content=content
           )
    )

def create_assistant_reply_content(
    sender:str, 
    content:str, 
    recipient:str="User", 
    dialogue_id:str=DEFAULT_GUID
    ) -> MessagePydantic:
    """
    Create an assistant reply chunk message.
    
    Args:
        sender (str): The sender of the message.
        chunk_no (int): The chunk number.
        chunk (str): The content of the chunk.
        content (str): The content of the message.
        recipient (str): The recipient of the message, defaults to "User".
        dialogue_id (str): The dialogue ID, defaults to "default-guid".
        
    Returns:
        MessagePydantic: A message object with type "reply".
    """
    return MessagePydantic(
       header=MessageHeaderPydantic(
           sender=sender,recipient=recipient
           ),
       body=ReplyBodyPydantic(
            content=content,
            dialogue_id=dialogue_id
            )
    )
    
def create_assistant_reply_chunk(
    sender:str, 
    chunk_no:int, 
    chunk:str, 
    content:str, 
    recipient:str="User", 
    dialogue_id:str=DEFAULT_GUID
    ) -> MessagePydantic:
    """
    Create an assistant reply chunk message.
    
    Args:
        sender (str): The sender of the message.
        chunk_no (int): The chunk number.
        chunk (str): The content of the chunk.
        content (str): The content of the message.
        recipient (str): The recipient of the message, defaults to "User".
        dialogue_id (str): The dialogue ID, defaults to "default-guid".
        
    Returns:
        MessagePydantic: A message object with type "reply".
    """
    return MessagePydantic(
       header=MessageHeaderPydantic(
           sender=sender,recipient=recipient
           ),
       body=ReplyBodyPydantic(
            content=content,
            dialogue_id=dialogue_id,
            chunk_no=chunk_no,
            chunk=chunk
            )
    )

def json(list:list[MessagePydantic]) -> str:
    """
    Convert a list of messages to JSON format.
    
    Args:
        list (list[MessagePydantic]): A list of messages to convert.
        
    Returns:
        str: A JSON string representation of the messages.
    """
    import json
    return json.dumps([message.model_dump() for message in list],indent=4)

def unjson(json_str:str) -> list[MessagePydantic]:
    """
    Convert a JSON string to a list of messages.
    
    Args:
        json_str (str): A JSON string representation of messages.
        
    Returns:
        list[MessagePydantic]: A list of messages.
    """
    import json as json_lib
    return [MessagePydantic.model_validate(message) for message in json_lib.loads(json_str)]    

def extract_recap(messages: list[MessagePydantic], last_n: int, max_recap_size: int) -> str:
    """
    Extract a recap of the last N messages, constrained by max_recap_size.

    Args:
        messages (list[MessagePydantic]): The full message history.
        last_n (int): Number of most recent messages to consider.
        max_recap_size (int): Maximum character length for the recap.

    Returns:
        str: A recap string containing recent messages up to the size limit.
    """
    # Step 1: Get the last N messages
    recent_messages = messages[-last_n:]

    # Step 2: Convert messages to simple role-content format
    simple_messages = convert_to_chat_messages(recent_messages)

    # Step 3: Accumulate until reaching max_recap_size
    recap_lines = []
    total_len = 0
    for msg in simple_messages:
        line = f"{msg['role'].capitalize()}: {msg['content'].strip()}"
        if total_len + len(line) > max_recap_size:
            break
        recap_lines.append(line)
        total_len += len(line)

    return "\n".join(recap_lines)
