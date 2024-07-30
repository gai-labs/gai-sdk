from gai.ttt.server.singleton_host import SingletonHost
from gai.lib.common.utils import free_mem
from gai.lib.common import utils
import os

def test_exllamav2_is_loaded_correctly():
    # arrange
    config = {
        "type": "ttt",
        "generator_name": "exllamav2-mistral7b",
        "engine": "gai.ttt.server.GaiExLlamaV2",
        "model_path": "models/exllamav2-mistral7b",
        "model_basename": "model",
        "max_seq_len": 8192,
        "prompt_format": "mistral",
        "hyperparameters": {
            "temperature": 0.85,
            "top_p": 0.8,
            "top_k": 50,
            "max_new_tokens": 1000,
        },
        "tool_choice": "auto",
        "max_retries": 5,
        "stop_conditions": ["<s>", "</s>", "user:","\n\n"],
        "no_flash_attn":True,
        "seed": None,
        "decode_special_tokens": False,
    }
    with SingletonHost.GetInstanceFromConfig(config) as host:

        # act: without AI placeholder
        try:
            # The following initialization will take place each time a completion is created
            host.generator.load_generator()
            host.generator.initialize_job_state(messages=[{"role":"user","content":"hello world"}],stream=False)
            host.generator.load_job()
        except Exception as e:
            # assert: should throw error if ai placeholder is missing
            assert "Last message should be an AI placeholder" in str(e)

        # act: with AI placeholder
        host.generator.load_generator()
        host.generator.initialize_job_state(messages=[{"role":"user","content":"hello world"},{"role":"assistant","content":""}],stream=False)
        host.generator.load_job()

        # assert
        exllamav2=host.generator

        # assert: config
        assert exllamav2.exllama_config.max_seq_len == 8192
        assert exllamav2.exllama_config.no_flash_attn == True
        assert exllamav2.exllama_config.model_dir == os.path.expanduser("~/.gai/models/exllamav2-mistral7b")

        # assert: cache
        assert exllamav2.cache.max_seq_len == 8192

        # assert: settings
        assert exllamav2.job.gen_settings.temperature == 0.85
        assert exllamav2.job.gen_settings.top_k == 50
        assert exllamav2.job.gen_settings.top_p == 0.8
        assert exllamav2.job.max_new_tokens == 1000
        assert "<s>" in exllamav2.job.stop_strings
        assert "</s>" in exllamav2.job.stop_strings
        assert "user:" in exllamav2.job.stop_strings
        assert "\n\n" in exllamav2.job.stop_strings
        assert exllamav2.job.decode_special_tokens == False

    free_mem()

def test_call_tool():
    config = {
        "type": "ttt",
        "generator_name": "exllamav2-mistral7b",
        "engine": "gai.ttt.server.GaiExLlamaV2",
        "model_path": "models/exllamav2-mistral7b",
        "model_basename": "model",
        "max_seq_len": 8192,
        "prompt_format": "mistral",
        "hyperparameters": {
            "temperature": 0.85,
            "top_p": 0.8,
            "top_k": 50,
            "max_new_tokens": 1000,
        },
        "tool_choice": "auto",
        "max_retries": 5,
        "stop_conditions": ["<s>", "</s>", "user:","\n\n"],
        "no_flash_attn":True,
        "seed": None,
        "decode_special_tokens": False,
    }
    with SingletonHost.GetInstanceFromConfig(config) as completions:
        messages = [
            {"role":"user","content":"What is the current time in Singapore?"},
            {"role":"assistant","content":""}
        ]
        tool_choice="required"
        tools = [
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
        ]
        response = completions.create(
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            stream=False)
        print(response)
        assert response.choices[0].finish_reason == "tool_calls"
        assert response.choices[0].message.tool_calls[0].type == "function"
        assert response.choices[0].message.tool_calls[0].function.name == "google"
        assert response.choices[0].message.tool_calls[0].function.arguments == '{"search_query": "current time Singapore"}'

