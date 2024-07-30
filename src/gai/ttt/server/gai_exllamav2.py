import os,torch,gc,json
from typing import Optional
from gai.lib.common.utils import get_app_path
from gai.lib.common.logging import getLogger
logger = getLogger(__name__)
from jsonschema import validate, ValidationError
from gai.lib.common.profile_function import profile_function
from gai.ttt.server.builders import CompletionsFactory

def correct_single_quote_json(s):
    rstr = ""
    escaped = False

    # Remove single quotes at the beginning and end of the string
    s=s.strip("'")

    for c in s:
    
        if c == "'" and not escaped:
            c = '"' # replace single with double quote
        
        elif c == "'" and escaped:
            rstr = rstr[:-1] # remove escape character before single quotes
        
        elif c == '"':
            c = '\\' + c # escape existing double quotes

        escaped = (c == "\\") # check for an escape character
        rstr += c # append the correct json
    
    return rstr

class GaiExLlamav2:

    def __init__(self, gai_config):
        if (gai_config is None):
            raise Exception("ExLlama_TTT2: gai_config is required")
        if gai_config.get("model_path",None) is None:
            raise Exception("ExLlama_TTT2: model_path is required")
        if gai_config.get("model_basename",None) is None:
            raise Exception("ExLlama_TTT2: model_basename is required")

        self.gai_config = gai_config
        self.model_dir = os.path.join(get_app_path(
        ), gai_config["model_path"])
        self.cache = None
        self.model = None
        self.tokenizer = None
        self.prompt = None
        self.generator = None
    
    def __enter__(self):
        self.load()
        return self
    
    def __exit__(self,exc_type, exc_value,traceback):
        self.unload()
        gc.collect()
        torch.cuda.empty_cache()


    @profile_function
    def load_config(self):
        from exllamav2 import ExLlamaV2Config
        config=ExLlamaV2Config()
        config.model_dir = self.model_dir
        config.prepare()
        config.max_seq_len = self.gai_config.get("max_seq_len",8192)
        config.no_flash_attn = self.gai_config.get("no_flash_attn",True)
        self.exllama_config = config

    @profile_function
    def load_model(self):
        from exllamav2 import ExLlamaV2
        self.model = ExLlamaV2(self.exllama_config)
    
    @profile_function    
    def load_cache(self):
        from exllamav2.cache import ExLlamaV2Cache_Q4
        self.cache = ExLlamaV2Cache_Q4(self.model, 
                                       lazy=True, 
                                       max_seq_len=self.gai_config.get("max_seq_len",8192))
        self.model.load_autosplit(self.cache)

    @profile_function
    def load_tokenizer(self):
        from exllamav2 import ExLlamaV2Tokenizer
        self.tokenizer = ExLlamaV2Tokenizer(self.exllama_config)

    @profile_function
    def load_generator(self):
        from exllamav2.generator import ExLlamaV2DynamicGenerator

        # Note: Dynamic Generator requires flash-attn 2.5.7+ to use paged attention and only supports Ampere GPUs or newer, otherwise set paged=False
        self.generator=ExLlamaV2DynamicGenerator(model=self.model, 
                                            cache=self.cache, 
                                            tokenizer=self.tokenizer,
                                            paged=False)
        self.generator.warmup()

    # initial load
    def load(self):
        self.load_config()
        self.load_model()
        self.load_cache()
        self.load_tokenizer()

    def unload(self):
        if self.generator is not None:
            del self.model
            del self.cache
            del self.tokenizer
            del self.generator
            import gc,torch
            gc.collect()
            torch.cuda.empty_cache()
        return self

    # Configure JSON Schema enforcement for tool call and response model
    def prepare_filters(self, tools, tool_choice, response_model):
        """
        1. tool_choice always takes precedence over schema.
           If schema is required, then tool_choice must be set to "none".
        2. If tool_choice is "auto", then tools schema will always be used but tools will include { "type": "text", "text": "..." }.
        3. If tool_choice is "required", then tools schema will always be used and tools will not include { "type": "text", "text": "..." }.
        4. If tool_choice is "none", then schema will be used if it is available.
        5. If tool_choice is "none" and schema is not available then output will always be text.
        6. If tool_choice is "none" and schema is available then output will always be based on schema, aka. JSON mode.        
        """

        self.validation_schema=None

        # Create filter from schema
        def create_filter(validation_schema,tokenizer):
            from lmformatenforcer.integrations.exllamav2 import build_token_enforcer_tokenizer_data
            tokenizer_data = build_token_enforcer_tokenizer_data(tokenizer)                
            from lmformatenforcer import JsonSchemaParser
            parser = JsonSchemaParser(validation_schema)
            from lmformatenforcer.integrations.exllamav2 import ExLlamaV2TokenEnforcerFilter
            from exllamav2.generator.filters.prefix import ExLlamaV2PrefixFilter
            return [ExLlamaV2TokenEnforcerFilter(parser, tokenizer_data),
                ExLlamaV2PrefixFilter(self.model, self.tokenizer, ["{","\n\n{"])]

        if not tools and tool_choice=="required":
            raise Exception("tool_choice='required' requires tools to be provided.")
        
        if (not self.is_validation_required(self.job_state)):
            self.validation_schema=None
            return None        

        if (self.is_using_tools(self.job_state)):
            # If toolcall is used, use toolcall schema and create filters for it
            from gai.lib.common.generators_utils import get_tools_schema
            self.validation_schema=get_tools_schema()
            return create_filter(validation_schema=self.validation_schema, tokenizer=self.tokenizer)        

        if (self.is_using_response_model(self.job_state)):
            # If user_defined schema, apply user_defined schema
            self.validation_schema=response_model
            return create_filter(validation_schema=self.validation_schema, tokenizer=self.tokenizer)

    # Configure intermediate prompts for tool call and response model formatted for the underlying LLM
    def prepare_prompt(self, messages, response_model, tools, tool_choice, stream):
        formatted_messages = messages

        if (self.is_using_tools(self.job_state)):

            # Add system_message for tool_call
            from gai.lib.common.generators_utils import apply_tools_message
            formatted_messages = apply_tools_message(messages=formatted_messages,
                tools=tools,
                tool_choice=tool_choice)
            
        if (self.is_using_response_model(self.job_state)):

            # Add system_message to use schema (response model) if tool_call are not applicable
            from gai.lib.common.generators_utils import apply_schema_prompt
            formatted_messages = apply_schema_prompt(messages=formatted_messages,
                schema=response_model)

        prompt_format = self.gai_config.get("prompt_format")
        from gai.lib.common.generators_utils import format_list_to_prompt
        return format_list_to_prompt(messages=formatted_messages, format_type=prompt_format, stream=stream)
    
    def is_validation_required(self,job_state):
        return self.is_using_tools(job_state) or self.is_using_response_model(job_state)

    # tools are only used when tool_choice is "required" and tools are available
    def is_using_tools(self,job_state):
        using_tools = (job_state["tool_choice"]=="required" and job_state["tools"])
        return using_tools

    # response model is only used when tool_choice is none and response_model is available
    def is_using_response_model(self,job_state):
        using_response_model = ((job_state["tool_choice"]=="none" or job_state["tool_choice"]=="auto") and job_state["response_model"])
        return using_response_model

    @profile_function
    def generate(self):

        eos = False
        while not eos:
            results = self.generator.iterate()
            for result in results:
                eos=result.get("eos",False)
        return result

    # Retry completions if validation is required
    def _generate_with_retries(self):
        retries=self.job_state["max_retries"]

        # Validation not required
        if not self.is_validation_required(self.job_state):
            return self.generate()
        
        # Validation required
        while retries>0:
            result=self.generate()
            try:
                if (self.is_using_tools(self.job_state)):
                    jsoned={
                        "type":"function",
                        "function":json.loads(result["full_completion"])
                    }
                    validate(instance=jsoned, schema=self.validation_schema)

                if (self.is_using_response_model(self.job_state)):
                    jsoned=json.loads(result["full_completion"])
                    validate(instance=jsoned, schema=self.validation_schema)

                # We are safe once we reach here since its either text or we have passed validations
                return result
            except ValidationError as e:
                logger.debug(f"GaiExLlamav2.generate_with_retries: Failed to validate {jsoned}.")
                retries-=1
                self.load_job()
            except Exception as e:
                logger.error(f"GaiExLlamav2.generate_with_retries: error={e}")
                raise e
        raise Exception("GaiExLlamav2.generate_with_retries: Validation of schema is required and failed after max retries.")

    def _streaming(self):
        eos = False
        result = None
        completed = ""

        if self.is_validation_required(self.job_state):
            raise Exception("GaiExLlamav2._streaming: Validation of schema is required and not supported in streaming mode.")

        while not eos:

            # Run one iteration of the generator. Returns a list of results
            results = self.generator.iterate()
            for result in results:
                text=result.get("text","")
                if text:
                    completed+=text
                    yield text
                eos=result.get("eos",False)

        yield result

    def load_job(self):

        from exllamav2.generator import ExLlamaV2Sampler
        settings = ExLlamaV2Sampler.Settings()
        settings.temperature=self.job_state["temperature"]
        settings.top_k=self.job_state["top_k"]
        settings.top_p=self.job_state["top_p"]
        settings = settings

        # Prepare settings.filters
        from exllamav2.generator import ExLlamaV2DynamicJob
        settings.filters = self.prepare_filters(tools=self.job_state["tools"], 
                                                     tool_choice=self.job_state["tool_choice"],
                                                     response_model=self.job_state["response_model"])
        if settings.filters:
            logger.info(f"GaiExLlamav2.load_job: apply filters.")
            settings.temperature=0
        prompt=self.prepare_prompt(messages=self.job_state["messages"],
                                   response_model=self.job_state["response_model"],
                                   tools=self.job_state["tools"],
                                   tool_choice=self.job_state["tool_choice"],
                                   stream=True)

        # Reset generator and run job
        self.job = ExLlamaV2DynamicJob(    
            input_ids = self.tokenizer.encode(prompt),
            gen_settings = settings,
            max_new_tokens = self.job_state["max_new_tokens"],
            completion_only = True,
            token_healing = True,
            seed = 1,
            stop_conditions=self.job_state["stop_conditions"],
            add_bos=False,
            #encode_special_tokens=self.job_state["encode_special_tokens"],
            decode_special_tokens=self.job_state["decode_special_tokens"],
        )
        # for pending in self.generator.pending_jobs:
        #     self.generator.cancel(pending)
        self.generator.enqueue(self.job)
        return self.generator               

    # Main driver for completions that calls _generate and _streaming,
    # but only return pure dictionary as output
    # Parameters are captured into a job_state for completions.
    def _create(self):
        try:
            self.load_job()
        except Exception as e:
            logger.error(f"GaiExLlamav2._create: Error loading job. error={e}")

        if not self.job_state["stream"]:
            return self._generate_with_retries()
        return (chunk for chunk in self._streaming())

    def initialize_job_state(self,
        messages:list,
        stream:Optional[bool], 
        tools:Optional[list]=None,
        tool_choice:Optional[str]=None,
        response_model=None,
        max_new_tokens:Optional[int]=None,
        stop_conditions:Optional[list]=None,
        temperature:Optional[float]=None,
        top_p:Optional[float]=None,
        top_k:Optional[int]=None,
        max_retries:int=None
        ):
        logger.info("gai_exllamav2.create: Initializing job_state.")
        if self.tokenizer.eos_token_id not in self.gai_config["stop_conditions"]:
            self.gai_config["stop_conditions"].append(self.tokenizer.eos_token_id)
        self.job_state = {
            "messages": messages,
            "stream": stream,
            "tools": tools,
            "response_model": response_model,
            "tool_choice": tool_choice or self.gai_config["tool_choice"],
            "stop_conditions": stop_conditions or self.gai_config["stop_conditions"],
            "prompt_format": self.gai_config["prompt_format"],
            "max_new_tokens": max_new_tokens or self.gai_config["hyperparameters"]["max_new_tokens"],
            "temperature": temperature or self.gai_config["hyperparameters"]["temperature"],
            "top_p": top_p or self.gai_config["hyperparameters"]["top_p"],
            "top_k": top_k or self.gai_config["hyperparameters"]["top_k"],  
            "max_retries": max_retries or self.gai_config["max_retries"],
            "decode_special_tokens": self.gai_config["decode_special_tokens"]
        }
        if self.is_validation_required(self.job_state):
            self.job_state["temperature"]=0        

    def create(self, 
        messages:list,
        stream:Optional[bool], 
        tools:Optional[list]=None,
        tool_choice:Optional[str]=None,
        response_model=None,
        max_new_tokens:Optional[int]=None,
        stop_conditions:Optional[list]=None,
        temperature:Optional[float]=None,
        top_p:Optional[float]=None,
        top_k:Optional[int]=None,
        max_retries:int=None
        ):
        self.load_generator()

        self.initialize_job_state(
            messages=messages,
            stream=stream,
            tools=tools,
            tool_choice=tool_choice,
            response_model=response_model,
            max_new_tokens=max_new_tokens,
            stop_conditions=stop_conditions,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_retries=max_retries
        )

        # Create completions
        response=self._create()

        # Convert Output
        factory = CompletionsFactory()        
        if not stream:
            if self.is_using_tools(self.job_state):
                logger.info(f"ExLlama_TTT2: factory.message.build_toolcall(response)")
                return factory.message.build_toolcall(response)
            else:
                logger.info(f"ExLlama_TTT2: factory.message.build_content(response)")
                return factory.message.build_content(response)
        else:
            return (chunk for chunk in factory.chunk.build_stream(response))



