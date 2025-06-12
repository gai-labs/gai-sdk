from ...types import (
    ChatCompletionChunk, 
    ChunkChoice, 
    ChoiceDelta, 
    ChoiceDeltaToolCall, 
    ChoiceDeltaToolCallFunction
    )

from gai.lib.logging import getLogger
logger = getLogger(__name__)

from datetime import datetime
from uuid import uuid4
from ..output_chunk_builder_base import OutputChunkBuilderBase

class OutputChunkBuilder(OutputChunkBuilderBase):

    def build_content(self,result,generator_name="ollama"):
        if (result is None):
            return None

        finish_reason = result.done_reason
        content = result.message.content

        if (content is None):
            return OutputChunkBuilder(
                ).add_chunk(generator=generator_name
                    ).add_chunk_choice_delta(finish_reason=None, role="assistant"
                        ).add_chunk_choice_delta_content(content=''
                            ).build()

        if (type(content) is str):
            return OutputChunkBuilder(
                ).add_chunk(generator=generator_name
                    ).add_chunk_choice_delta(finish_reason=None, role=None
                        ).add_chunk_choice_delta_content(content=content
                            ).build()

        if (finish_reason):
            return OutputChunkBuilder(
                ).add_chunk(generator=generator_name
                    ).add_chunk_choice_delta(finish_reason=finish_reason, role=None
                        ).build()
