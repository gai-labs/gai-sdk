from .output_message_builder import OutputMessageBuilder
from .output_chunk_builder import OutputChunkBuilder

class CompletionsFactory:

    def __init__(self):
        self.message = OutputMessageBuilder()
        self.chunk = OutputChunkBuilder()