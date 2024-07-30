import threading
from gai.lib.common import logging, generators_utils
import os
from dotenv import load_dotenv
load_dotenv()
logger = logging.getLogger(__name__)
from gai.ttt.server.gai_exllamav2 import GaiExLlamav2

class SingletonHost:
    __instance = None       # singleton

    @staticmethod
    def GetInstanceFromPath(generator_name,config_path=None):
        """Static method to access this singleton class's instance."""
        if SingletonHost.__instance == None:
            config_path=os.path.expanduser(config_path)
            config = generators_utils.load_generators_config(config_path)
            SingletonHost.__instance=SingletonHost(config[generator_name])
        return SingletonHost.__instance

    @staticmethod
    def GetInstanceFromConfig(config):
        """Static method to access this singleton class's instance."""
        if SingletonHost.__instance == None:
            SingletonHost.__instance=SingletonHost(config)
        return SingletonHost.__instance

    def __init__(self,config):
        """Virtually private constructor."""
        if SingletonHost.__instance is not None:
            raise Exception(
                "SingletonHost: This class is a singleton! Access using GetInstance().")
        else:
            # This class only has 4 attributes

            # config is always the first to be loaded from constructor
            self.config = config

            # generator is loaded by calling load()
            self.generator = None

            # generator_name matches config["generator_name"] and is loaded only when self.generator is successfully loaded 
            self.generator_name = None

            # Finally, for thread safety and since this is run locally, use semaphore to ensure only one thread can access the generator at a time
            self.semaphore = threading.Semaphore(1)

            SingletonHost.__instance = self

    def __enter__(self):
        self.load()
        return self
    
    def __exit__(self,exc_type, exc_value,traceback):
        self.unload()
        import gc,torch
        gc.collect()
        torch.cuda.empty_cache()

    # This is idempotent
    def load(self):

        if self.generator_name == self.config["generator_name"]:
            logger.debug(
                "SingletonHost.load: Generator is already loaded. Skip loading.")
            return self

        if self.generator_name and self.generator_name != self.config["generator_name"]:
            logger.debug(
                "SingletonHost.load: New generator_name specified, unload current generator.")
            if self.generator:
                self.unload()
            return self

        if self.config["generator_name"] == "exllamav2-mistral7b" and self.config["type"] == "ttt":
            try:
                target_name=self.config["generator_name"]
                logger.info(f"SingletonHost: Loading generator {target_name}...")
                self.generator = GaiExLlamav2(gai_config=self.config)
                self.generator.load()
                self.generator_name=target_name
                return self
            except Exception as e:
                self.unload()
                logger.error(
                    f"SingletonHost: Error loading generator {self.generator_name}: {e}")
                raise e
            
    def unload(self):
        if self.generator is not None:
            self.generator.unload()
            del self.generator
            self.generator = None
            self.generator_name = None
            import gc,torch
            gc.collect()
            torch.cuda.empty_cache()
        return self

    def create(self, **model_params):
        if self.generator is None:
            logger.error("SingletonHost.create: Generator is not loaded.")
            raise Exception("SingletonHost.create: Generator is not loaded.")
        with self.semaphore:
            return self.generator.create(**model_params)

    def token_count(self, text):
        if self.generator is None:
            logger.error("SingletonHost.create: Generator is not loaded.")
            raise Exception("SingletonHost.create: Generator is not loaded.")
        if hasattr(self.generator, 'token_count'):
            return self.generator.token_count(text)
        raise Exception("token_count is not supported by this generator.")

    def get_token_ids(self, text):
        if self.generator is None:
            logger.error("SingletonHost.create: Generator is not loaded.")
            raise Exception("SingletonHost.create: Generator is not loaded.")
        if hasattr(self.generator, 'get_token_ids'):
            return self.generator.get_token_ids(text)
        raise Exception("get_token_ids is not supported by this generator.")

    async def index_async(self, 
                          collection_name, 
                          file_path, 
                          file_type=None,
                          title='', 
                          source= '', 
                          abstract='',
                          authors='',
                          publisher ='',
                          published_date='', 
                          comments='',
                          keywords='',
                          chunk_size=None, 
                          chunk_overlap=None, 
                          status_publisher=None):
        if self.generator is None:
            logger.error("SingletonHost.create: Generator is not loaded.")
            raise Exception("SingletonHost.create: Generator is not loaded.")

        if self.generator.generator_name != "rag":
            logger.error(
                f"SingletonHost.index: The generator {self.generator.generator_name} does not support indexing.")
            raise Exception(
                f"SingletonHost.index: The generator {self.generator.generator_name} does not support indexing.")
        with self.semaphore:
            return await self.generator.index_async(
                collection_name=collection_name, 
                file_path=file_path,
                file_type=file_type,
                title=title,
                source=source,
                abstract=abstract,
                authors=authors,
                publisher=publisher,
                published_date=published_date,
                comments=comments,
                keywords=keywords,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                status_publisher=status_publisher)

    def retrieve(self, collection_name, query_texts, n_results=None):
        if self.generator is None:
            logger.error("SingletonHost.create: Generator is not loaded.")
            raise Exception("SingletonHost.create: Generator is not loaded.")

        if self.generator.generator_name != "rag":
            logger.error(
                f"SingletonHost.retrieve: The generator {self.generator.generator_name} does not support retrieval.")
            raise Exception(
                f"SingletonHost.retrieve: The generator {self.generator.generator_name} does not support retrieval.")
        with self.semaphore:
            return self.generator.retrieve(collection_name, query_texts, n_results)

