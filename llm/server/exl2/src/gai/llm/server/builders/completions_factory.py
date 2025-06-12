from gai.llm.server.builders import OutputMessageBuilder
from gai.llm.server.builders import OutputChunkBuilder

class CompletionsFactory:

    def __init__(self):
        self.message = OutputMessageBuilder()
        self.chunk = OutputChunkBuilder()


