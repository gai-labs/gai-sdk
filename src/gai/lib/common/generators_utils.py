import json
import os,re
from typing import List
from gai.lib.common.utils import get_gai_config

# A simple utility to validate if all items in model params are in the whitelist.
def validate_params(model_params,whitelist_params):
    for key in model_params:
        if key not in whitelist_params:
            raise Exception(f"Invalid param '{key}'. Valid params are: {whitelist_params}")

# A simple utility to filter items in model params that are also in the whitelist.
def filter_params(model_params,whitelist_params):
    filtered_params={}
    for key in model_params:
        if key in whitelist_params:
            filtered_params[key]=model_params[key]
    return filtered_params

# A simple utility to load generators config.
def load_generators_config(file_path=None):
    return get_gai_config(file_path)["gen"]

# This is used to compress a list into a smaller string to be passed as a single USER message to the prompt template.
def chat_list_to_string(messages):
    if type(messages) is str:
        return messages
    prompt=""        
    for message in messages:
        if prompt:
            prompt+="\n"
        content = message['content'].strip()
        role = message['role'].strip()
        if content:
            prompt += f"{role}: {content}"
        else:
            prompt += f"{role}:"
    return prompt

# This is useful for converting text dialog to chatgpt-style dialog
def chat_string_to_list(messages):
    # Split the messages into lines
    lines = messages.split('\n')

    # Prepare the result list
    result = []

    # Define roles
    roles = ['system', 'user', 'assistant']

    # Initialize current role and content
    current_role = None
    current_content = ''

    # Process each line
    for line in lines:
        # Check if the line starts with a role
        for role in roles:
            if line.lower().startswith(role + ':'):
                # If there is any content for the current role, add it to the result
                if current_role is not None and current_content.strip() != '':
                    result.append({'role': current_role, 'content': current_content.strip()})
                
                # Start a new role and content
                current_role = role
                current_content = line[len(role) + 1:].strip()
                break
        else:
            # If the line does not start with a role, add it to the current content
            current_content += ' ' + line.strip()

    # Add the last role and content to the result
    if current_role is not None:
        result.append({'role': current_role, 'content': current_content.strip()})

    return result

def chat_list_to_INST(input_list):
    # Initialize an empty string for the output
    output = "<s>\n\t[INST]\n"
    
    # if last message is an AI placeholder, remove it
    last_role = input_list[-1]["role"].lower()
    last_content = input_list[-1]["content"]
    if last_role != "system" and last_role != "user" and last_content == "":
        input_list.pop()

    # Loop through the list of dictionaries
    for item in input_list:
        # Check the role
        role = item["role"].lower()
        if role == "system":
            # Add the system message
            output += f"\t\t<<SYS>>\n\t\t\t{item['content']}\n\t\t<</SYS>>\n"
        elif role == "user":
            # Add the user message
            output += f"\t\t{item['content']}\n"
            output += "\t[/INST]\n\n\t"
        else:
            # Add the AI message
            output += f"{item['content']}\n\n"
            # AI message marks the end of 1 turn
            output += "</s>\n"
            # Add the beginning of next turn
            output += "<s>\n\t[INST]\n"
   
    return output

def INST_output_to_output(output_string):
    # The rfind method returns the last index where the substring is found
    last_index = output_string.rfind('[/INST]\n\n\t')

    # Add the length of '[/INST]\n\n\t' to get the start of the desired substring
    start_of_substring = last_index + len('[/INST]\n\n\t')

    # Extract the substring from start_of_substring till the end of the string
    result = output_string[start_of_substring:]

    return result

def ASSISTANT_output_to_output(output_string):
    return re.split('\n.+:',output_string)[-1].strip()

def has_ai_placeholder(messages):
    message = messages[-1]
    if message["role"].lower() != "system" and message["role"].lower() != "user" and message["content"] == "":
        return True
    return False

async def word_streamer_async( char_generator):
    buffer = ""
    async for byte_chunk in char_generator:
        if type(byte_chunk) == bytes:
            byte_chunk = byte_chunk.decode("utf-8", "replace")
        buffer += byte_chunk
        words = buffer.split(" ")
        if len(words) > 1:
            for word in words[:-1]:
                yield word
                yield " "
            buffer = words[-1]
    yield buffer            

def word_streamer( char_generator):
    buffer = ""
    for chunk in char_generator:
        if chunk:
            if type(chunk) == bytes:
                chunk = chunk.decode("utf-8", "replace")
            buffer += chunk
            words = buffer.split(" ")
            if len(words) > 1:
                for word in words[:-1]:
                    yield word
                    yield " "
                buffer = words[-1]
    yield buffer


def get_tools_schema():
    return {
        "type": "object",
        "properties": {
            "function": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string"
                    },
                    "arguments": {
                        "type": "object",
                    }
                },
                "required": ["name", "arguments"]
            },
        },
        "required": ["function"]            
    }

def apply_tools_message( messages: List, tools:dict, tool_choice:str):
    # Proceed only if tools are available
    if not tools:
        return messages

    # Check if tools are required and add a tools prompt
    if tools:
        if tool_choice == "none":
            # If tool_choice == "none", skip adding tools
            return messages
        
        if tool_choice == "required":
            # Create a system message to introduce the tools
            system_message = {"role":"system","content":
            """
            1. Select the most probable tool from the list below that is most suitable for responding to the user's message and respond only in JSON and nothing else.

                {tools}

            2. You must return one tool from the list and must not respond with anything else.
            """}   

        # When tool_choice == "auto", the system can return a tool response
        # or a text response.
        if tool_choice == "auto":

            # Create a system message to introduce the tools
            system_message = {"role":"system","content":
            """
            1. Review the <tools> below and assess if any of them is suitable for responding to the user's message.

                {tools}

            2. If none of the tools are suitable, you can respond with a <text> response that looks like the following:
                
            {{
                "function": {{
                    "name": "text",
                    "arguments": {{
                        "text": "This is a text response."
                    }}
                }}
            }}
            """}

        # Create system_message
        system_message["content"] = system_message["content"].format(
            tools=tools)

        # Insert the system message immediately before the last user_message.                
        ai_placeholder = None
        if has_ai_placeholder(messages):
            ai_placeholder = messages.pop()
        user_message = messages.pop()
        messages.append(system_message)
        messages.append(user_message)
        if ai_placeholder:
            messages.append(ai_placeholder)

    return messages

def apply_schema_prompt( messages: List, schema):

    # Apply schema. Note that tool schema will override any provided schema.
    if schema:
        system_message={"role":"system","content":f"""Begin your response with an open curly brace. Your response must be parseable by this json schema:
        ```jsonschema
        {schema}
        ```"""}
        #system_message={"role":"system","content":f"You will respond to the user's message based only on the following JSON schema {schema}. Begin your response with a curly bracket '{{' and end it with a curly bracket '}}'."}

        # Insert the system message immediately before the last user_message.                
        ai_placeholder = None
        if has_ai_placeholder(messages):
            ai_placeholder = messages.pop()
        user_message = messages.pop()
        messages.append(system_message)
        messages.append(user_message)
        if ai_placeholder:
            messages.append(ai_placeholder)

    return messages

def format_list_to_prompt(messages, format_type="llama3",stream=False):
    prompt=""
    if messages[-1]["content"]:
        raise Exception("Last message should be an AI placeholder")
    if format_type == "llama3":
        prompt="<|begin_of_text|>"
        for message in messages:
            role = message['role']
            role_prompt=f"<|start_header_id|>{role}<|end_header_id|>"
            content = message['content']
            if content:
                prompt+=f"{role_prompt}\n\n{content}<|eot_id|>"
            else:
                prompt+=role_prompt
        return prompt

    if format_type == "mistral":
        # I might be paranoid but somehow it seemed that different prompt format may be required for stream vs generation.

        if stream:
            prompt="<s>"
            for message in messages:
                role = message['role']
                content = message['content']
                if content:
                    prompt+=f"{role}: {content}\n"
                    if role.lower() == "assistant":
                        prompt+="</s><s>"
                else:
                    prompt+=f"{role}:"
            return prompt

        if not stream:
            # According to bartowski, the prompt format is: <s>[INST]  {prompt} [/INST]</s> and doesn't support system message.
            prompt="<s>"
            for message in messages:
                role = message['role']
                content = message['content']
                if role.lower() == "system" or role.lower() == "user":
                    if content:
                        prompt+=f"[INST]   {content}  [/INST]"
                if role.lower() == "assistant":
                    if content:
                        prompt+=f"{content}</s><s>"
            prompt.replace("[/INST][INST]","")
            return prompt

    raise Exception(f"Invalid format type '{format_type}'")