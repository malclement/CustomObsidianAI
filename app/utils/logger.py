import json
import logging
import sys
import traceback
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any
from typing import Dict
from typing import Optional
from typing import Union

from fastapi import Request

# Create logs directory if it doesn't exist
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

# Configure the root logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

# Create a custom logger
logger = logging.getLogger("obsidian_api")
logger.setLevel(logging.INFO)

# Create handlers
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)

# Create file handlers for different log types
info_file_handler = logging.FileHandler(log_dir / "app.log")
info_file_handler.setLevel(logging.INFO)

error_file_handler = logging.FileHandler(log_dir / "errors.log")
error_file_handler.setLevel(logging.ERROR)

request_file_handler = logging.FileHandler(log_dir / "requests.log")
request_file_handler.setLevel(logging.INFO)

# Create formatters and add them to handlers
log_format = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
json_format = logging.Formatter("%(message)s")

console_handler.setFormatter(log_format)
info_file_handler.setFormatter(log_format)
error_file_handler.setFormatter(log_format)
request_file_handler.setFormatter(json_format)

# Add handlers to the logger
logger.addHandler(console_handler)
logger.addHandler(info_file_handler)
logger.addHandler(error_file_handler)


class RequestLogger:
    """Custom logger for API requests and responses"""

    def __init__(self):
        self.logger = logging.getLogger("obsidian_api.requests")
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(request_file_handler)
        self._request_ids = {}

    def get_request_id(self, request: Request) -> str:
        """Get or create a unique ID for a request"""
        client = (
            f"{request.client.host}:{request.client.port}"
            if request.client
            else "unknown"
        )
        if client not in self._request_ids:
            self._request_ids[client] = str(uuid.uuid4())
        return self._request_ids[client]

    def log_request(
        self,
        request: Request,
        body: Any = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Log an incoming request"""
        request_id = self.get_request_id(request)

        # Create log entry
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "request_id": request_id,
            "method": request.method,
            "url": str(request.url),
            "client_ip": request.client.host if request.client else "unknown",
            "client_port": request.client.port if request.client else "unknown",
            "type": "request",
        }

        # Add body if provided (and not too large)
        if body:
            try:
                # Create a deep copy to avoid modifying the original
                if isinstance(body, dict):
                    # Handle content truncation safely
                    safe_body = {}
                    for k, v in body.items():
                        if k == "content" and isinstance(v, str) and len(v) > 500:
                            safe_body[k] = v[:500] + "... [TRUNCATED]"
                        elif k == "messages" and isinstance(v, list):
                            safe_messages = []
                            for msg in v:
                                safe_msg = dict(msg)
                                if (
                                    "content" in safe_msg
                                    and isinstance(safe_msg["content"], str)
                                    and len(safe_msg["content"]) > 500
                                ):
                                    safe_msg["content"] = (
                                        safe_msg["content"][:500] + "... [TRUNCATED]"
                                    )
                                safe_messages.append(safe_msg)
                            safe_body[k] = safe_messages
                        else:
                            safe_body[k] = v
                    log_entry["body"] = safe_body
                else:
                    log_entry["body"] = (
                        str(body)[:500] + "... [TRUNCATED]"
                        if len(str(body)) > 500
                        else body
                    )
            except Exception as e:
                log_entry["body"] = f"[Error serializing body: {str(e)}]"

        # Add query params if provided
        if params:
            log_entry["params"] = params

        try:
            self.logger.info(json.dumps(log_entry))
        except Exception as e:
            self.logger.error(f"Error logging request: {str(e)}")

        return request_id

    def log_response(
        self,
        request_id: str,
        status_code: int,
        response_body: Any = None,
        processing_time: Optional[float] = None,
    ) -> None:
        """Log an outgoing response"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "request_id": request_id,
            "status_code": status_code,
            "type": "response",
        }

        # Add processing time if provided
        if processing_time is not None:
            log_entry["processing_time_ms"] = processing_time

        # Add response body if provided (and not too large)
        if response_body:
            try:
                # For dictionaries, we need to be careful about modification
                if isinstance(response_body, dict):
                    # Make a shallow copy first
                    safe_body = {}

                    # Handle choices specifically for OpenAI response format
                    if "choices" in response_body:
                        safe_body = response_body.copy()  # Shallow copy
                        safe_choices = []

                        for choice in response_body["choices"]:
                            safe_choice = choice.copy()  # Shallow copy of each choice

                            if "message" in choice and "content" in choice["message"]:
                                # Make a copy of message to avoid modifying original
                                safe_message = choice["message"].copy()
                                orig_content = choice["message"]["content"]

                                if len(orig_content) > 500:
                                    safe_message["content"] = (
                                        orig_content[:500] + "... [TRUNCATED]"
                                    )

                                safe_choice["message"] = safe_message

                            safe_choices.append(safe_choice)

                        safe_body["choices"] = safe_choices
                    else:
                        # For regular dictionaries, copy key-value pairs
                        for k, v in response_body.items():
                            if isinstance(v, str) and len(v) > 500:
                                safe_body[k] = v[:500] + "... [TRUNCATED]"
                            else:
                                safe_body[k] = v

                    log_entry["body"] = safe_body
                else:
                    # For non-dictionary types, just truncate if needed
                    content_str = str(response_body)
                    if len(content_str) > 500:
                        log_entry["body"] = content_str[:500] + "... [TRUNCATED]"
                    else:
                        log_entry["body"] = response_body
            except Exception as e:
                log_entry["body"] = f"[Error serializing response: {str(e)}]"

        try:
            self.logger.info(json.dumps(log_entry))
        except Exception as e:
            self.logger.error(f"Error logging response: {str(e)}")

    def log_error(
        self,
        request_id: str,
        status_code: int,
        error: Union[str, Exception],
        detailed: bool = True,
    ) -> None:
        """Log an error response"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "request_id": request_id,
            "status_code": status_code,
            "type": "error",
        }

        # Add error message
        if isinstance(error, Exception):
            log_entry["error"] = str(error)
            log_entry["error_type"] = error.__class__.__name__

            # Add traceback if detailed logging is enabled
            if detailed:
                log_entry["traceback"] = traceback.format_exc()
        else:
            log_entry["error"] = error

        try:
            self.logger.info(json.dumps(log_entry))
        except Exception as e:
            self.logger.error(f"Error logging error: {str(e)}")

        logger.error(
            f"Request {request_id} failed with status {status_code}: {log_entry.get('error', 'Unknown error')}"
        )


# Create a singleton instance
request_logger = RequestLogger()


def get_logger():
    """Get the application logger"""
    return logger


def get_request_logger():
    """Get the request logger"""
    return request_logger
