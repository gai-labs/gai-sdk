from fastapi import FastAPI, Body
from pydantic import BaseModel, Field
from gai.lib.constants import DEFAULT_GUID
from gai.lib.tests import get_pyproject_path, get_pyproject_version

pyproject_path = get_pyproject_path()
version_no = get_pyproject_version(pyproject_path)
app = FastAPI(title="gai-sdk", description="gai-sdk API", version=version_no)


@app.get("/")
async def version():
    return {
        "app": app.title,
        "version": app.version,
        "description": app.description,
    }


class CreateDialogueRequest(BaseModel):
    user_id: str = Field(description="User GUID", default=DEFAULT_GUID)


@app.post("/dialogue/create")
async def create_dialogue(request: CreateDialogueRequest = Body(default=None)):
    user_id = DEFAULT_GUID
    dialogue_id = DEFAULT_GUID
    if request:
        user_id = request.user_id
    from gai.messages import FileDialogue
    dialogue = FileDialogue(caller_id=user_id, dialogue_id=dialogue_id)
    dialogue.reset()
    return {
        "user_id": user_id,
        "dialogue_id": dialogue_id,
        "message": "Dialogue created successfully"
    }


def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
