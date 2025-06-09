import os
import sys
import time
import toml
import asyncio
import threading
import multiprocessing as mp
from typing import Any, Dict, Optional, Union

from dotenv import load_dotenv
load_dotenv()

from rich.console import Console
console = Console()

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

# Internal modules
from gai.lib.diagnostics import free_mem
from gai.lib.color import red, green
from gai.lib.config import GaiConfig, GaiGeneratorConfig, GaiToolConfig, config_helper
from gai.lib.logging import getLogger
logger = getLogger(__name__)

from gai.lib.errors import ModelNotFoundException

def get_app_version(pyproject_toml):
    with open(pyproject_toml) as f:
        pyproject = toml.load(f)
    return pyproject["project"]["version"]

def get_pyproject(file:Optional[str]=None):
    """
    The argument is expecting __file__ from the calling module so that it can traverse up the directory tree to find the pyproject.toml file.
    If the file is not provided, it will use the current working directory.
    """
    if file is None:
        cwd = os.getcwd()    
    else:
        cwd = os.path.dirname(file)
    while cwd != "/":
        if os.path.exists(os.path.join(cwd, "pyproject.toml")):
            cwd = os.path.join(cwd, "pyproject.toml")
            break
        cwd = os.path.dirname(cwd)
    if not os.path.exists(cwd):
        raise FileNotFoundError("pyproject.toml not found.")
    return cwd

def get_project_name(pyproject_toml):
    with open(pyproject_toml) as f:
        pyproject = toml.load(f)
    return pyproject["project"]["name"]

# This tells fastapi which path to host the swagger ui page.
def get_swagger_url():
    swagger_url=None
    if "SWAGGER_URL" in os.environ and os.environ["SWAGGER_URL"]:
        swagger_url=os.environ["SWAGGER_URL"]
        logger.info(f"swagger={swagger_url}")
    else:
        logger.info("swagger_url=disabled.")
    return swagger_url

def configure_cors(app: FastAPI):
    allowed_origins_str = "*"
    if "CORS_ALLOWED" in os.environ:
        allowed_origins_str = os.environ["CORS_ALLOWED"]    # from .env
    allowed_origins = allowed_origins_str.split(",")  # split the string into a list
    logger.info(f"allow_origins={allowed_origins}")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

def configure_semaphore():
    use_semaphore = os.getenv("USE_SEMAPHORE", "False").lower() == "true"
    semaphore = None
    if use_semaphore:
        logger.info("Using semaphore")
        import asyncio
        semaphore = asyncio.Semaphore(2)
    return semaphore

async def acquire_semaphore(semaphore):
    while semaphore:
        try:
            await asyncio.wait_for(semaphore.acquire(), timeout=0.1)
            break
        except asyncio.TimeoutError:
            logger.warn("_streaming: Server is busy")
            await asyncio.sleep(1)

def release_semaphore(semaphore):
    if semaphore:
        semaphore.release()
        logger.debug("_streaming: Server is available")

def get_startup_event(
    app,
    category: str,
    pyproject_toml: str,
    generator_config: Union[GaiGeneratorConfig|GaiToolConfig],
):
    async def startup_event():
        
        try:
            # check freemem before loading the model
            #free_mem()

            # version check
            logger.info(f"Starting Gai LLM Service ({category}) {get_app_version(pyproject_toml)}")
            
            # extract the default generator config for a category and add it to the app state

            app.state.generator_config = generator_config

            # initialize host if it is a GaiGeneratorConfig
            if isinstance(generator_config, GaiGeneratorConfig):
            
                host = SingletonHost.GetInstanceFromConfig(generator_config)
                host.load()
                logger.info(f"Model loaded = [{generator_config.name}]")
                app.state.host = host

                # check freemem after loading the model
                free_mem()    
        except Exception as e:
            logger.error(f"Failed to load default model: {e}")
            raise e

    return startup_event

def get_shutdown_event(app):
    
    async def shutdown_event():
        host = app.state.host
        if host:
            host.unload()

    return shutdown_event

def get_generator_config(request: Request) -> GaiGeneratorConfig:
    """
    Dependency to grab the GaiGeneratorConfig you stored in app.state
    """
    return request.app.state.generator_config

def get_generator(request: Request):
    return request.app.state.host.generator

def create_app(pyproject_toml:str, category:str, generator_config: GaiGeneratorConfig):
    
    """
    A helper function to create a FastAPI app with CORS and startup/shutdown events.
    """

    app=FastAPI(
        title="Gai Generators Service",
        description="""Gai Generators Service""",
        version=get_app_version(pyproject_toml),
        docs_url=get_swagger_url()
        )
    configure_cors(app)

    # Event Handlers
    app.add_event_handler("startup", get_startup_event(app, category=category, pyproject_toml=pyproject_toml, generator_config=generator_config))
    app.add_event_handler("shutdown", get_shutdown_event(app))

    return app

def create_tool_app(pyproject_toml:str, category:str, tool_config: GaiToolConfig):
    
    """
    A helper function to create a FastAPI app with CORS and startup/shutdown events.
    """

    app=FastAPI(
        title="Gai MCP Service",
        description="""Gai MCP Service""",
        version=get_app_version(pyproject_toml),
        docs_url=get_swagger_url()
        )
    configure_cors(app)

    return app

class _GeneratorWorker(mp.Process):
    def __init__(self,
                 config: GaiGeneratorConfig,
                 cmd_queue: mp.Queue,
                 res_queue: mp.Queue):
        super().__init__(daemon=True)
        self.config = config
        self.cmd_queue = cmd_queue
        self.res_queue = res_queue

    def run(self):
        from gai.chat.server.gai_exllamav2 import GaiExLlamav2

        # 1) load model
        try:
            self.worker = GaiExLlamav2(self.config)
            self.worker.load()
            self.res_queue.put(("loaded", None))
        except Exception as e:
            self.res_queue.put(("error", f"load failed: {e}"))
            return

        # 2) handle commands
        while True:
            cmd, payload = self.cmd_queue.get()
            if cmd == "create":
                try:
                    if payload.get("stream", False):
                        # streaming: iterate and push each chunk
                        for chunk in self.worker.create(**payload):
                            self.res_queue.put(("stream", chunk))
                        # signal end-of-stream
                        self.res_queue.put(("stream_end", None))
                    else:
                        # non-streaming: get single result
                        result = self.worker.create(**payload)
                        self.res_queue.put(("result", result))
                except Exception as e:
                    self.res_queue.put(("error", f"create failed: {e}"))

            elif cmd == "exit":
                # teardown
                self.worker.unload()
                self.res_queue.put(("exited", None))
                break


class SingletonHost:
    __instance: Optional["SingletonHost"] = None

    @staticmethod
    def GetInstanceFromConfig(config: GaiGeneratorConfig, verbose=True):
        if SingletonHost.__instance is None:
            SingletonHost.__instance = SingletonHost(config, verbose)
        else:
            SingletonHost.__instance.config = config
            SingletonHost.__instance.verbose = verbose
        return SingletonHost.__instance

    def __init__(self, config: GaiGeneratorConfig, verbose=True):
        if SingletonHost.__instance is not None:
            raise RuntimeError("Use GetInstanceFromConfig()")
        self.config = config
        self.verbose = verbose

        self._model_timeout = 180
        self._cmd_queue: mp.Queue = mp.Queue()
        self._res_queue: mp.Queue = mp.Queue()
        self._proc: Optional[_GeneratorWorker] = None
        self.generator_name: Optional[str] = None

    def load(self):

        if self._proc and self._proc.is_alive() and self.generator_name == self.config.name:
            return

        if self._proc and self._proc.is_alive():
            self.unload()
            time.sleep(1)

        logger.info(f"SingletonHost.load: spawning worker for {self.config.name!r}")
        self._proc = _GeneratorWorker(self.config, self._cmd_queue, self._res_queue)
        self._proc.start()

        try:
            tag, payload = self._res_queue.get(timeout=self._model_timeout)
        except Empty:
            raise RuntimeError("Timeout after 180s: loading model timed out.")
        
        if tag == "error":
            raise RuntimeError(payload)
        self.generator_name = self.config.name
        logger.info(f"SingletonHost.load: self.generator_name={self.generator_name}")

    def create(self, model:str, **kwargs) -> Any:
        """
        This function creates a out-of-process generator for the given model.
        If the model is different from the currently loaded model, it will unload the current generator and load the new one.
        
        model must be provided so that the generator can be loaded or switched.
        This parameter is only used by the host and not passed to the generator since the generator will run whatever model is loaded by the host.
        
        If kwargs.get('stream', False) is True, return a generator that yields chunks.
        Otherwise, block and return the final result.
        """
        
        # Step 1: Check if the model is loaded

        config = config_helper.get_generator_config(name=model)
        if config is None:
            raise ModelNotFoundException(model)
        if config.name != self.generator_name:
            
            # Do not compare with model name because model can be an alias.
            # Use the model name to load the config then compare the config's name with self.generator_name.
            
            logger.info(f"SingletonHost.create: unloading {self.generator_name!r} and loading {model!r}")
            self.unload()
            self.config = config
            self.load()
        
        # Step 2: Check if the generator is loaded
        
        if not self._proc or not self._proc.is_alive():
            raise RuntimeError("Generator not loaded")

        # Step 3: Prepare the request

        stream = kwargs.get("stream", False)

        # dispatch the request
        self._cmd_queue.put(("create", kwargs))

        if not stream:
            tag, payload = self._res_queue.get()
            if tag == "result":
                return payload
            else:
                raise RuntimeError(payload)

        # streaming path: return a Python generator
        def _stream_gen():
            while True:
                tag, payload = self._res_queue.get()
                if tag == "stream":
                    yield payload
                elif tag == "stream_end":
                    return
                else:  # error
                    raise RuntimeError(payload)

        return _stream_gen()

    def unload(self):
        if not self._proc:
            return

        self._cmd_queue.put(("exit", None))
        tag, _ = self._res_queue.get(timeout=30)
        # join/terminate child
        self._proc.join(timeout=10)
        if self._proc.is_alive():
            self._proc.terminate()
            self._proc.join()

        self._proc = None
        self.generator_name = None

        logger.info("SingletonHost.unload: Generator unloaded.")


