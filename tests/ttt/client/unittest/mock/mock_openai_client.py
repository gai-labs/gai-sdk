import json,os,sys
from gai.lib.common.utils import this_dir
this_path = this_dir(__file__)
from openai.types.chat.chat_completion import ChatCompletion
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk

class MockOpenAI:

    class MockCompletions:

        def openai_create(self,**kwargs):
            tool_choice = kwargs.get("tool_choice","auto")
            stream = kwargs.get("stream",False)
            messages = kwargs.get("messages")
            
            if tool_choice == "none" or messages[0].get("content") == "Tell me a one paragraph story":
                if not stream:
                    filename="1a_generate_text_openai.json"
                    fullpath=os.path.join(this_path, filename) 
                    with open(fullpath,"r") as f:
                        try:
                            content = f.read()
                            jsoned = json.loads(content)
                            response=ChatCompletion(**jsoned)
                            response.extract= lambda: response.choices[0].message.content
                            return response
                        except Exception as e:
                            print(e)
                            raise(e)
                else:
                    def streamer():
                        filename="1b_stream_text_openai.json"
                        fullpath=os.path.join(this_path, filename) 
                        with open(fullpath,"r") as f:
                            list = json.load(f)
                            for chunk in list:
                                chunk = ChatCompletionChunk(**chunk)
                                chunk.extract = lambda: chunk.choices[0].delta.content
                                yield chunk
                    return (chunk for chunk in streamer())

        def gai_create(self,**kwargs):
            tool_choice = kwargs.get("tool_choice","auto")
            stream = kwargs.get("stream",False)
            messages = kwargs.get("messages")
            
            if tool_choice == "none" or messages[0].get("content") == "Tell me a one paragraph story":
                if not stream:
                    filename="2a_generate_text_gai.json"
                    fullpath=os.path.join(this_path, filename) 
                    with open(fullpath,"r") as f:
                        try:
                            content = f.read()
                            jsoned = json.loads(content)
                            response=ChatCompletion(**jsoned)
                            response.extract= lambda: response.choices[0].message.content
                            return response
                        except Exception as e:
                            print(e)
                            raise(e)
                else:
                    def streamer():
                        filename="2b_stream_text_gai.json"
                        fullpath=os.path.join(this_path, filename) 
                        with open(fullpath,"r") as f:
                            list = json.load(f)
                            for chunk in list:
                                chunk = ChatCompletionChunk(**chunk)
                                chunk.extract = lambda: chunk.choices[0].delta.content
                                yield chunk
                    return (chunk for chunk in streamer())
                
        def create(self, **kwargs):
            model = kwargs.get("model")
            if model == "gpt-4":
                response = self.openai_create(**kwargs)
            else:
                response = self.gai_create(**kwargs)
            return response

    class MockChatCompletions:
        def __init__(self):
            self.completions = MockOpenAI.MockCompletions()


    def __init__(self):
        self.chat = MockOpenAI.MockChatCompletions()
