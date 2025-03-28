import time
import traceback
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
from fastapi import Request
from fastapi import status
from pydantic import BaseModel
from pydantic import Field
from pydantic import validator

from app.services.completion_service import CompletionService
from app.services.model_service import ModelService
from app.utils.dependencies import get_completion_service
from app.utils.dependencies import get_model_service
from app.utils.error_handlers import ModelIncompatibleException
from app.utils.error_handlers import ModelNotFoundException
from app.utils.error_handlers import RateLimitExceededException
from app.utils.error_handlers import TokenLimitExceededException
from app.utils.logger import get_logger
from app.utils.logger import get_request_logger

router = APIRouter()
logger = get_logger()
request_logger = get_request_logger()


# Models matching OpenAI's request/response format
class Message(BaseModel):
    role: str = Field(
        ..., description="The role of the message author (system, user, assistant)"
    )
    content: str = Field(..., description="The content of the message")

    @validator("role")
    def validate_role(cls, v):
        allowed_roles = ["system", "user", "assistant"]
        if v not in allowed_roles:
            raise ValueError(f"Role must be one of: {', '.join(allowed_roles)}")
        return v


class CompletionRequest(BaseModel):
    model: str = Field(..., description="ID of the model to use")
    messages: List[Message] = Field(
        ..., description="A list of messages comprising the conversation so far"
    )
    temperature: Optional[float] = Field(
        0.7, description="What sampling temperature to use", ge=0.0, le=2.0
    )
    top_p: Optional[float] = Field(
        1.0,
        description="An alternative to sampling with temperature, called nucleus sampling",
        ge=0.0,
        le=1.0,
    )
    max_tokens: Optional[int] = Field(
        None, description="The maximum number of tokens to generate", ge=1, le=4096
    )
    stream: Optional[bool] = Field(
        False, description="Whether to stream back partial progress"
    )

    @validator("messages")
    def validate_messages(cls, v):
        if not any(msg.role == "user" for msg in v):
            raise ValueError("At least one message must have role 'user'")
        return v


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


@router.post("/chat/completions", response_model=CompletionResponse)
async def create_completion(
    request: Request,
    req_body: CompletionRequest = Body(...),
    model_service: ModelService = Depends(get_model_service),
    completion_service: CompletionService = Depends(get_completion_service),
) -> CompletionResponse:
    """
    OpenAI API-compatible chat completion endpoint

    This endpoint follows the same input/output format as OpenAI's chat completion API,
    but uses Hugging Face models under the hood.
    """
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    start_time = time.time()

    try:
        logger.info(f"Processing completion request for model: {req_body.model}")

        if req_body.stream:
            raise ModelIncompatibleException(
                model_name=req_body.model,
                operation="streaming",
                message="Streaming is not supported yet",
            )

        # Extract user content - assuming the last user message is what we want to process
        user_messages = [msg for msg in req_body.messages if msg.role == "user"]
        if not user_messages:
            raise ValueError("No user message found in the request")

        user_content = user_messages[-1].content

        # Get system messages - in OpenAI format, usually the first message is system
        system_messages = [msg for msg in req_body.messages if msg.role == "system"]
        system_prompt = system_messages[0].content if system_messages else None

        # Generate completion
        try:
            completion_text, usage = await completion_service.generate_completion(
                model_name=req_body.model,
                user_content=user_content,
                system_prompt=system_prompt,
                temperature=req_body.temperature,
                top_p=req_body.top_p,
                max_tokens=req_body.max_tokens,
            )
        except Exception as e:
            # Check for specific error types and convert to appropriate exceptions
            error_message = str(e).lower()

            if "not found" in error_message or "model_not_found" in error_message:
                raise ModelNotFoundException(req_body.model)
            elif "rate limit" in error_message:
                raise RateLimitExceededException()
            elif (
                "token limit" in error_message
                or "maximum context length" in error_message
            ):
                raise TokenLimitExceededException(
                    token_count=len(user_content.split()) * 2,  # Rough estimate
                    max_tokens=req_body.max_tokens or 4096,
                )
            else:
                logger.error(
                    f"Error generating completion: {str(e)}\n{traceback.format_exc()}"
                )
                raise e

        # Format the response to match OpenAI's format
        response = CompletionResponse(
            id=f"chatcmpl-{uuid.uuid4().hex}",
            created=int(time.time()),
            model=req_body.model,
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

        # Log successful completion
        process_time = (time.time() - start_time) * 1000
        logger.info(
            f"Completion successful for model {req_body.model} in {process_time:.2f}ms. "
            f"Tokens: {usage['total_tokens']}"
        )

        return response

    except (
        ModelNotFoundException,
        ModelIncompatibleException,
        TokenLimitExceededException,
        RateLimitExceededException,
    ) as e:
        # These exceptions are handled by custom exception handlers
        raise

    except ValueError as e:
        # Handle validation errors
        logger.warning(f"Validation error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail={"error": str(e)}
        )

    except Exception as e:
        # Handle unexpected errors
        logger.error(f"Unexpected error: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "An unexpected error occurred during processing"},
        )
