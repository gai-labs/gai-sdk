from gai.lib.common import file_utils
from gai.lib.common.logging import getLogger
logger = getLogger(__name__)
import os
from nltk.tokenize import sent_tokenize

class Chunker:

    @staticmethod
    def split(text, chunk_size=None, chunk_overlap=None):
        chunk_hash = file_utils.create_chunk_id_base64(text)
        chunks=[]
        try:
            if chunk_size is None:
                chunk_size = 2000
            if chunk_overlap is None:
                chunk_overlap = 200
            chunks_dir = file_utils.get_chunk_dir("/tmp",chunk_hash)
            file_utils.split_text(text=text, 
                chunks_dir=chunks_dir,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap)
            chunks = os.listdir(chunks_dir)
            return chunks
        except Exception as error:
            logger.error(f"Chunker.split: Failed to split chunks. error={error}")
            raise error

    @staticmethod
    def sentences(text):
        return sent_tokenize(text)

