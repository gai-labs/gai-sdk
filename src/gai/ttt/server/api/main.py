import os
os.environ["LOG_LEVEL"]="DEBUG"
import uuid,json
from dotenv import load_dotenv
load_dotenv()

# WEB
from fastapi import FastAPI, Body
from pydantic import BaseModel
from typing import List, Optional
from fastapi.responses import StreamingResponse,JSONResponse
from fastapi.encoders import jsonable_encoder

# GAI
from gai.lib.common.errors import *
from gai.lib.common.logging import getLogger
logger = getLogger(__name__)
from gai.lib.common.utils import free_mem
from gai.ttt.server.singleton_host import SingletonHost

# Configure Dependencies
import dependencies
logger.info(f"Starting Gai Generators Service v{dependencies.APP_VERSION}")
logger.info(f"Version of gai_gen installed = {dependencies.LIB_VERSION}")
free_mem()

swagger_url = dependencies.get_swagger_url()
app=FastAPI(
    title="Gai Generators Service",
    description="""Gai Generators Service""",
    version=dependencies.APP_VERSION,
    docs_url=swagger_url
    )
dependencies.configure_cors(app)
semaphore = dependencies.configure_semaphore()


host = None
gai_config = None

# STARTUP
from gai.lib.common.utils import get_gai_config
DEFAULT_GENERATOR=os.getenv("DEFAULT_GENERATOR")
async def startup_event():
    global host,gai_config
    # Perform initialization here
    try:
        gai_config = get_gai_config()
        DEFAULT_GENERATOR = gai_config["gen"]["default"]["ttt"]
        if os.environ.get("DEFAULT_GENERATOR",None):
            DEFAULT_GENERATOR = os.environ.get("DEFAULT_GENERATOR")
        gai_config = gai_config["gen"][DEFAULT_GENERATOR]
        host = SingletonHost.GetInstanceFromConfig(gai_config)
        host.load()

        from gai.lib.common.color import Color
        color = Color()
        color.white(text=f"Default model loaded: {DEFAULT_GENERATOR}")
        free_mem()
    except Exception as e:
        logger.error(f"Failed to load default model: {e}")
        raise e
app.add_event_handler("startup", startup_event)

# SHUTDOWN
async def shutdown_event():
    # Perform cleanup here
    try:
        host.unload()
    except Exception as e:
        logger.error(f"Failed to unload default model: {e}")
        raise e
app.add_event_handler("shutdown", shutdown_event)

# GET /gen/v1/chat/version
@app.get("/gen/v1/chat/version")
async def version():
    return JSONResponse(status_code=200, content={
        "version": dependencies.APP_VERSION
    })

# POST /gen/v1/chat/completions
class MessageRequest(BaseModel):
    role: str
    content: str
class ChatCompletionRequest(BaseModel):
    messages: List[MessageRequest]
    stream: Optional[bool] = False
    tools: Optional[list] = None
    tool_choice: Optional[str] = None
    response_model: Optional[dict] = None    
    max_new_tokens: Optional[int] = None
    stop_conditions: Optional[list] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None

@app.post("/gen/v1/chat/completions")
async def _text_to_text(req: ChatCompletionRequest = Body(...)):
    response=None
    try:
        messages = req.messages
        response = host.create(
            messages=[message.model_dump() for message in messages],
            stream=req.stream,
            tools=req.tools,
            tool_choice=req.tool_choice,
            response_model=req.response_model,
            max_new_tokens=req.max_new_tokens,
            stop_conditions=req.stop_conditions,
            temperature=req.temperature,
            top_p=req.top_p,
            top_k=req.top_k
        )
        if req.stream:
            def streamer():
                for chunk in response:
                    try:
                        if chunk is not None:
                            chunk = jsonable_encoder(chunk)
                            chunk = json.dumps(chunk) + "\n"
                            yield chunk
                    except Exception as e:
                        logger.warn(f"Error in stream: {e}")
                        continue

            return StreamingResponse(streamer())
        else:
            return response
    except Exception as e:
        if (str(e)=='context_length_exceeded'):
            raise ContextLengthExceededException()
        if (str(e)=='model_service_mismatch'):
            raise GeneratorMismatchException()
        id=str(uuid.uuid4())
        logger.error(str(e)+f" id={id}")
        raise InternalException(id)

if __name__ == "__main__":
    import uvicorn

    config = uvicorn.Config(
        app=app, 
        host="0.0.0.0", 
        port=12031, 
        timeout_keep_alive=180,
        timeout_notify=150,
        workers=1
    )
    server = uvicorn.Server(config=config)
    server.run()
