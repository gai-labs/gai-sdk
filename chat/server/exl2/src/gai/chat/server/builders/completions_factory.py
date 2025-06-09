from gai.chat.server.builders import OutputMessageBuilder
from gai.chat.server.builders import OutputChunkBuilder

class CompletionsFactory:

    def __init__(self):
        self.message = OutputMessageBuilder()
        self.chunk = OutputChunkBuilder()


