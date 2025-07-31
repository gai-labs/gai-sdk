import os
import json
from fastapi import FastAPI, Body, Request, APIRouter
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, Field
from typing import Optional

from gai.asm.agents import ToolUseAgent, PendingUserInputError, AutoResumeError
from gai.mcp.client.mcp_client import McpAggregatedClient
from gai.messages.monologue import FileMonologue
from gai.messages.dialogue import FileDialogue
from gai.lib.constants import DEFAULT_GUID
from gai.lib.config import config_helper
from gai.lib.tests import get_pyproject_path, get_pyproject_version
from gai.lib.logging import getLogger

logger = getLogger(__name__)

PYPROJECT_PATH = get_pyproject_path()
VERSION_NO = get_pyproject_version(PYPROJECT_PATH)
MONOLOGUE_PATH = "~/.gai/logs/{agent_name}.log"
DEFAULT_MODEL_NAME = "sonnet-4"
agent_cache = {}


# Define a streaming response to yield chunks of data
async def streamer(resp):
    """
    Returns an async generator for LLM response stream
    """
    async for chunk in resp:
        if isinstance(chunk, str):
            yield chunk
        else:
            tool = {}
            if isinstance(chunk, list):
                for item in chunk:
                    if item.get("name"):
                        tool["name"] = item["name"]
                    if item.get("input"):
                        inputs = item.get("input", {})
                        if isinstance(inputs, dict):
                            for key, value in inputs.items():
                                if isinstance(value, str):
                                    if len(value) > 100:
                                        inputs[key] = value[:100] + "..."
                                    else:
                                        inputs[key] = value
                        tool["input"] = inputs
                yield json.dumps(tool)


async def precheck_streamer(gen):
    """
    Stream LLM response and check if the agent is expecting user input for resume or if task has completed
    """
    try:
        first = await anext(gen)  # force the first item to detect any errors early

        async def yield_with_first():
            yield first
            async for chunk in gen:
                yield chunk

        return StreamingResponse(yield_with_first(), media_type="application/json")

    except PendingUserInputError as e:
        logger.error(f"Pending user input error: {e}")
        return JSONResponse(
            status_code=409,
            content={"error": "Pending user input", "details": str(e)},
        )
    except AutoResumeError as e:
        logger.error(f"Task has completed error: {e}")
        return JSONResponse(
            status_code=409, content={"error": "Task has completed", "details": str(e)}
        )
    except Exception as e:
        logger.error(f"Error during streaming: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": "Streaming failed", "details": str(e)},
        )


def make_agent(model_name: str, agent_name: str, mcp_names: Optional[list[str]] = None):
    if agent_cache.get(agent_name):
        agent = agent_cache[agent_name]
    else:
        llm_config = config_helper.get_client_config(model_name)
        file_path = os.path.expanduser(
            MONOLOGUE_PATH.format(
                agent_name=agent_name.replace(" ", "_"),
            )
        )
        monologue = FileMonologue(agent_name=agent_name, file_path=file_path)
        aggregated_client = McpAggregatedClient(mcp_names if mcp_names else [])
        agent = ToolUseAgent(
            agent_name=agent_name,
            llm_config=llm_config,
            aggregated_client=aggregated_client,
            monologue=monologue,
        )
        agent_cache[agent_name] = agent
    return agent


router = APIRouter()


class CreateDialogueRequest(BaseModel):
    user_id: str = Field(description="User GUID", default=DEFAULT_GUID)


@router.post("/dialogue/create")
async def create_dialogue(request: CreateDialogueRequest = Body(default=None)):
    user_id = DEFAULT_GUID
    dialogue_id = DEFAULT_GUID
    if request:
        user_id = request.user_id
    dialogue = FileDialogue(caller_id=user_id, dialogue_id=dialogue_id)
    dialogue.reset()
    return {
        "user_id": user_id,
        "dialogue_id": dialogue_id,
        "message": "Dialogue created successfully",
    }


class ASMStartRequest(BaseModel):
    agent_name: str = Field(description="Agent name", default="AgentX")
    model_name: str = Field(description="Model name", default="sonnet-4")
    mcp_names: list[str] = Field(
        description="List of MCP names",
        default=["mcp-pseudo", "mcp-filesystem", "mcp-web"],
    )
    user_message: str = Field(description="User message to start the ASM")
    dialogue_id: str = Field(description="Dialogue ID", default=DEFAULT_GUID)


@router.post("/asm/start")
async def asm_start(req: ASMStartRequest = Body(...)):
    agent = make_agent(
        model_name=DEFAULT_MODEL_NAME,
        agent_name=req.agent_name,
        mcp_names=req.mcp_names,
    )

    # Create a dialogue and extract recap
    dialogue = FileDialogue(caller_id=DEFAULT_GUID, dialogue_id=req.dialogue_id)
    recap = dialogue.extract_recap()

    # Start the agent asynchronously
    resp = agent.start(user_message=req.user_message, recap=recap)
    return await precheck_streamer(streamer(resp))


class ASMResumeRequest(BaseModel):
    agent_name: str = Field(description="Agent name", default="AgentX")
    model_name: str = Field(description="Model name", default="sonnet-4")
    mcp_names: list[str] = Field(
        description="List of MCP names",
        default=["mcp-pseudo", "mcp-filesystem", "mcp-web"],
    )
    user_message: Optional[str] = Field(
        description="User message to resume the conversation", default=None
    )


@router.post("/asm/resume")
async def asm_resume(req: ASMResumeRequest = Body(...)):
    agent = make_agent(
        model_name=DEFAULT_MODEL_NAME,
        agent_name=req.agent_name,
        mcp_names=req.mcp_names,
    )
    resp = agent.resume(user_message=req.user_message)
    return await precheck_streamer(streamer(resp))


class ASMUndoRequest(BaseModel):
    agent_name: str = Field(description="Agent name")


@router.post("/asm/undo")
async def asm_undo(req: ASMUndoRequest = Body(...)):
    agent = make_agent(
        model_name=DEFAULT_MODEL_NAME,
        agent_name=req.agent_name,
    )
    agent.undo()
    return agent.fsm.state


class GetMonologueRequest(BaseModel):
    agent_name: str = Field(description="Agent name")


@router.get("/asm/monologue")
async def asm_monologue(request: GetMonologueRequest = Body(...)):
    file_path = os.path.expanduser(
        MONOLOGUE_PATH.format(
            agent_name=request.agent_name.replace(" ", "_"),
        )
    )
    monologue = FileMonologue(agent_name=request.agent_name, file_path=file_path)
    return monologue.list_chat_messages()


def main():
    import uvicorn

    app = FastAPI(title="gai-sdk", description="gai-sdk API", version=VERSION_NO)

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request, exc: RequestValidationError
    ):
        # For logging 422 errors
        logger.error(f"Validation error for {request.url.path}: {exc.args}")
        return JSONResponse(
            status_code=422,
            content={
                "detail": exc.args,
                "body": exc.body,
            },
        )

    @app.get("/")
    async def version():
        return {
            "app": app.title,
            "version": app.version,
            "description": app.description,
        }

    app.include_router(router)
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
