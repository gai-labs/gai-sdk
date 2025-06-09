from unittest.mock import patch
from gai.lib.config import GaiGeneratorConfig
from openai.types.chat.chat_completion import ChatCompletion

mock_config = {
            "type": "ttt",
            "engine": "exllamav2",
            "model": "dolphin3.0_llama3.1:4.25bpw",
            "name": "dolphin3.0_llama3.1:4.25bpw:exl2",
            "hyperparameters": {
                "temperature": 0.85,
                "top_p": 0.8,
                "top_k": 50,
                "max_tokens": 1000,
                "tool_choice": "auto",
                "max_retries": 5,
                "stop": [
                    "<|im_end|>",
                    "</s>",
                    "[/INST]"
                ]
            },
            "extra": {
                "no_flash_attn": True,
                "seed": None,
                "decode_special_tokens": False,
                "model_path": "models/Dolphin3.0-Llama3.1-8B-4_25bpw-exl2",
                "max_seq_len": 8192,
                "prompt_format": "llama"
            },
            "module": {
                "name": "gai.chat.server.gai_exllamav2",
                "class": "GaiExLlamav2"
            }
        }

@patch("gai.chat.server.gai_exllamav2.GaiExLlamav2._create")
@patch("gai.chat.server.gai_exllamav2.GaiExLlamav2.initialize_job_state")
@patch("gai.chat.server.gai_exllamav2.GaiExLlamav2.load_generator")
@patch("gai.chat.server.gai_exllamav2.GaiExLlamav2.load")
def test_text_completion(mock_load,  mock_generator,mock_initialize_job_state,mock_create):
    mock_job_state = {'messages': [{'role': 'user', 'content': 'Tell me a one paragraph story'}, {'role': 'assistant', 'content': ''}], 'stream': False, 'tools': None, 'json_schema': None, 'tool_choice': 'auto', 'stop': ['<|im_end|>', '</s>', '[/INST]', 32768], 'prompt_format': 'mistral', 'max_tokens': 1000, 'temperature': 0.85, 'top_p': 0.8, 'top_k': 50, 'max_retries': 5, 'decode_special_tokens': False}
    mock_create.return_value={'job': None, 'stage': 'streaming', 'eos': True, 'serial': 1, 'eos_reason': 'stop_token', 'eos_triggering_token_id': 32768, 'eos_triggering_token_str': '<|im_end|>', 'full_completion': "A one-paragraph story: In a small, peaceful village, there lived a humble farmer named Tom. Despite his hard work in the fields, he was always kind and generous to everyone around him. One day, a wealthy nobleman visiting the village took notice of Tom's virtues and decided to reward him with a large sum of money. The news spread like wildfire, but Tom remained humble and thanked everyone for their kindness, reminding them that true happiness comes from within.", 'new_tokens': 101, 'prompt_tokens': 23, 'time_enqueued': 0.0005319118499755859, 'time_prefill': 0.08852362632751465, 'time_generate': 3.232391119003296, 'cached_pages': 0, 'cached_tokens': 0, 'held': {'text': '<|im_end|>', 'token_ids': None}}
    from gai.chat.server.gai_exllamav2 import GaiExLlamav2
    from gai.lib.config import config_helper
    config = config_helper.get_generator_config(generator_config=mock_config)
    host = GaiExLlamav2(config)
    host.load()
    host.job_state = mock_job_state
    
    response = host.create(
        messages=[{"role":"user","content":"Tell me a one paragraph story"},
                    {"role":"assistant","content":""}],
        stream=False)
    
    assert response.choices[0].finish_reason == "stop"
    assert response.choices[0].message.content == "A one-paragraph story: In a small, peaceful village, there lived a humble farmer named Tom. Despite his hard work in the fields, he was always kind and generous to everyone around him. One day, a wealthy nobleman visiting the village took notice of Tom's virtues and decided to reward him with a large sum of money. The news spread like wildfire, but Tom remained humble and thanked everyone for their kindness, reminding them that true happiness comes from within."

@patch("gai.chat.server.gai_exllamav2.GaiExLlamav2._create")
@patch("gai.chat.server.gai_exllamav2.GaiExLlamav2.initialize_job_state")
@patch("gai.chat.server.gai_exllamav2.GaiExLlamav2.load_generator")
@patch("gai.chat.server.gai_exllamav2.GaiExLlamav2.load")
def test_call_tool(mock_load,  mock_generator,mock_initialize_job_state,mock_create):
    mock_job_state = {'messages': [{'role': 'user', 'content': 'What is the current time in Singapore?'}, {'role': 'assistant', 'content': ''}], 'stream': False, 'tools': [{...}], 'json_schema': None, 'tool_choice': 'required', 'stop': ['<|im_end|>', '</s>', '[/INST]', 32768], 'prompt_format': 'mistral', 'max_tokens': 1000, 'temperature': 0, 'top_p': 0.8, 'top_k': 50, 'max_retries': 5, 'decode_special_tokens': False}
    mock_create.return_value={'job': None, 'stage': 'streaming', 'eos': True, 'serial': 1, 'eos_reason': 'stop_string', 'eos_triggering_string': '<|im_end|>', 'token_ids': None, 'full_completion': ' {\n    "function": {\n      "name": "google",\n      "arguments": {\n        "search_query": "current time in Singapore"\n      }\n    }\n  }', 'new_tokens': 50, 'prompt_tokens': 418, 'time_enqueued': 0.0006117820739746094, 'time_prefill': 0.2687973976135254, 'time_generate': 1.7001996040344238, 'cached_pages': 0, 'cached_tokens': 0, 'held': {'text': '<|im_end|>'}}
    from gai.chat.server.gai_exllamav2 import GaiExLlamav2
    from gai.lib.config import config_helper
    config = config_helper.get_generator_config(generator_config=mock_config)
    host = GaiExLlamav2(config)
    host.load()
    host.job_state = mock_job_state
    
    response = host.create(
        messages=[
            {"role":"user","content":"What is the current time in Singapore?"},
            {"role":"assistant","content":""}
        ],
        tool_choice="required",
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "google",
                    "description": "The 'google' function is a powerful tool that allows the AI to gather external information from the internet using Google search. It can be invoked when the AI needs to answer a question or provide information that requires up-to-date, comprehensive, and diverse sources which are not inherently known by the AI. For instance, it can be used to find current date, current news, weather updates, latest sports scores, trending topics, specific facts, or even the current date and time. The usage of this tool should be considered when the user's query implies or explicitly requests recent or wide-ranging data, or when the AI's inherent knowledge base may not have the required or most current information. The 'search_query' parameter should be a concise and accurate representation of the information needed.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "search_query": {
                                "type": "string",
                                "description": "The search query to search google with. For example, to find the current date or time, use 'current date' or 'current time' respectively."
                            }
                        },
                        "required": ["search_query"]
                    }
                }
            }
        ],
        stream=False)
    
    assert response.choices[0].finish_reason == "tool_calls"
    assert response.choices[0].message.tool_calls[0].type == "function"
    assert response.choices[0].message.tool_calls[0].function.name == "google"
    assert response.choices[0].message.tool_calls[0].function.arguments == '{"search_query": "current time in Singapore"}'
