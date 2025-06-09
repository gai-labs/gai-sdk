import os
import uuid
import uvicorn
from fastapi import APIRouter, Body, Depends
from fastapi.responses import JSONResponse, StreamingResponse
from gai.lib.logging import getLogger
logger = getLogger(__name__)

# GAI
from gai.chat.lib.dtos import ChatCompletionRequest, ModelDownloadRequest
from gai.lib.config import config_helper
from gai.lib.api import get_app_version, create_app, get_project_name, get_pyproject
from gai.lib.errors import ModelNotFoundException, ContextLengthExceededException, GeneratorMismatchException, InternalException

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

default_generator = os.environ.get("DEFAULT_GENERATOR")

### ----------------- create FastAPI app ----------------- ###

pyproject_toml = get_pyproject()
builtin_config_path = os.path.join(os.path.dirname(__file__),"../config/gai.yml")
generator_config = config_helper.get_and_update_generator_config(generator_name=default_generator, builtin_config_path=builtin_config_path)
app = create_app(pyproject_toml, category="ttt",generator_config=generator_config)

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    # For logging 422 errors
    logger.error(f"Validation error for {request.url.path}: {exc.errors()}")
    return JSONResponse(
        status_code=422,
        content={
            "detail": exc.errors(),
            "body": exc.body,
        },
    )


router = APIRouter()


# PONG

@router.get("/")
async def root():
    return get_project_name(pyproject_toml=pyproject_toml)

### ----------------- TTT ----------------- ###

# GET /gen/v1/chat/version
@router.get("/gen/v1/chat/version")
async def version():
    return JSONResponse(status_code=200, content={
        "version": get_app_version(pyproject_toml=pyproject_toml)
    })

# POST /gen/v1/chat/completions
@router.post("/gen/v1/chat/completions")
async def post_chat_completions(req: ChatCompletionRequest = Body(...)):
    host = app.state.host
    response=None
    try:
        messages = req.messages
        response = host.create(
            model=req.model,
            messages=[message.model_dump() for message in messages],
            stream=req.stream,
            tools=req.tools,
            tool_choice=req.tool_choice,
            json_schema=req.json_schema,
            max_tokens=req.max_tokens,
            stop=req.stop,
            temperature=req.temperature,
            top_p=req.top_p,
            top_k=req.top_k
        )
        if req.stream:
            def streamer():
                for chunk in response:
                    try:
                        if chunk is not None:
                            print(chunk.choices[0].delta.content, end="", flush=True)
                            chunk = chunk.json() + "\n"
                            yield chunk
                    except Exception as e:
                        logger.warn(f"Error in stream: {e}")
                        continue
            return StreamingResponse(streamer(), media_type="application/json")
        else:
            print(response.choices[0].message.content)
            return response

    except ModelNotFoundException as me:
        logger.error(f"main.process_stream: {str(me)}")
        raise
    except ContextLengthExceededException as cle:
        logger.error(f"main.process_stream: {str(cle)}")
        raise
    except GeneratorMismatchException as gme:
        logger.error(f"main.process_stream: {str(gme)}")
        raise
    except Exception as e:
        id=str(uuid.uuid4())
        logger.error(f"main.process_stream: {str(e)} id={id}")
        raise InternalException(id)

@router.post("/gen/v1/chat/pull")
async def post_chat_pull(req: ModelDownloadRequest = Body(...)):
    from gai.chat.lib.generators_utils import download, text_progress_callback
    download(name_or_config=req.model, status_callback=text_progress_callback)
    print("Download complete")

app.include_router(router, dependencies=[Depends(lambda: app.state.host)])

# __main__
if __name__ == "__main__":
    
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
