import time
import uuid
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

from fastapi import APIRouter
from fastapi import Body
from fastapi import Depends
from fastapi import HTTPException
from pydantic import BaseModel
from pydantic import Field

from app.services.completion_service import CompletionService
from app.services.model_service import ModelService
from app.utils.dependencies import get_completion_service
from app.utils.dependencies import get_model_service

router = APIRouter()


# Models matching OpenAI's request/response format
class Message(BaseModel):
    role: str = Field(
        ..., description="The role of the message author (system, user, assistant)"
    )
    content: str = Field(..., description="The content of the message")


class CompletionRequest(BaseModel):
    model: str = Field(..., description="ID of the model to use")
    messages: List[Message] = Field(
        ..., description="A list of messages comprising the conversation so far"
    )
    temperature: Optional[float] = Field(
        0.7, description="What sampling temperature to use"
    )
    top_p: Optional[float] = Field(
        1.0,
        description="An alternative to sampling with temperature, called nucleus sampling",
    )
    max_tokens: Optional[int] = Field(
        None, description="The maximum number of tokens to generate"
    )
    stream: Optional[bool] = Field(
        False, description="Whether to stream back partial progress"
    )


class CompletionChoice(BaseModel):
    index: int
    message: Message
    finish_reason: str


class CompletionUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class CompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[CompletionChoice]
    usage: CompletionUsage


@router.post("/chat/completions")
async def create_completion(
    request: CompletionRequest = Body(...),
    model_service: ModelService = Depends(get_model_service),
    completion_service: CompletionService = Depends(get_completion_service),
) -> CompletionResponse:
    """
    OpenAI API-compatible chat completion endpoint

    This endpoint follows the same input/output format as OpenAI's chat completion API,
    but uses Hugging Face models under the hood.
    """
    try:
        if request.stream:
            raise HTTPException(
                status_code=400, detail="Streaming is not supported yet"
            )

        # Extract user content - assuming the last user message is what we want to process
        user_messages = [msg for msg in request.messages if msg.role == "user"]
        if not user_messages:
            raise HTTPException(status_code=400, detail="No user message found")

        user_content = user_messages[-1].content

        # Get system messages - in OpenAI format, usually the first message is system
        system_messages = [msg for msg in request.messages if msg.role == "system"]
        system_prompt = system_messages[0].content if system_messages else None

        # Generate completion
        completion_text, usage = await completion_service.generate_completion(
            model_name=request.model,
            user_content=user_content,
            system_prompt=system_prompt,
            temperature=request.temperature,
            top_p=request.top_p,
            max_tokens=request.max_tokens,
        )

        # Format the response to match OpenAI's format
        response = CompletionResponse(
            id=f"chatcmpl-{uuid.uuid4().hex}",
            created=int(time.time()),
            model=request.model,
            choices=[
                CompletionChoice(
                    index=0,
                    message=Message(role="assistant", content=completion_text),
                    finish_reason="stop",
                )
            ],
            usage=CompletionUsage(
                prompt_tokens=usage["prompt_tokens"],
                completion_tokens=usage["completion_tokens"],
                total_tokens=usage["total_tokens"],
            ),
        )

        return response

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error generating completion: {str(e)}"
        )
