import copy
import json
import time
import uuid
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

from fastapi import Request
from fastapi import Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.base import RequestResponseEndpoint
from starlette.types import ASGIApp

from app.utils.logger import get_logger
from app.utils.logger import get_request_logger

logger = get_logger()
request_logger = get_request_logger()


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for logging requests and responses"""

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        # Generate request ID and store it in request state
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id

        # Get request body (if applicable)
        request_body = None
        if request.method in ["POST", "PUT", "PATCH"]:
            try:
                # Make a copy of the request body
                body_bytes = await request.body()
                request_body = await request.json() if body_bytes else None
                # Create a new body stream
                request._body = body_bytes
            except:
                # If we can't parse as JSON, that's ok
                pass

        # Get query parameters
        query_params = dict(request.query_params) if request.query_params else None

        # Log request
        request_logger.log_request(
            request=request, body=request_body, params=query_params
        )

        # Track timing
        start_time = time.time()

        # Process the request
        try:
            response = await call_next(request)

            # Calculate processing time
            process_time = (time.time() - start_time) * 1000

            # Log response (for non-streaming responses)
            if not response.headers.get("content-type", "").startswith(
                "text/event-stream"
            ):
                # Try to get response body for logging (if possible)
                response_body = None

                # We will NOT modify the response body for logging purposes
                # This avoids Content-Length mismatches

                # Just log the metadata without trying to read the body
                status_code = response.status_code
                content_type = response.headers.get("content-type", "unknown")
                content_length = response.headers.get("content-length", "unknown")

                log_response_info = {
                    "status_code": status_code,
                    "content_type": content_type,
                    "content_length": content_length,
                    "processing_time_ms": process_time,
                }

                # Log the response metadata only
                request_logger.log_response(
                    request_id=request_id,
                    status_code=status_code,
                    response_body=log_response_info,
                    processing_time=process_time,
                )

            # Add processing time header
            response.headers["X-Process-Time"] = f"{process_time:.2f}ms"
            # Add request ID header for tracking
            response.headers["X-Request-ID"] = request_id

            return response

        except Exception as e:
            # Log the error
            process_time = (time.time() - start_time) * 1000
            logger.error(f"Unhandled exception in middleware: {str(e)}")

            # Log error response
            request_logger.log_error(
                request_id=request_id, status_code=500, error=e, detailed=True
            )

            # Re-raise exception to let FastAPI exception handlers deal with it
            raise


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Simple rate limiting middleware"""

    def __init__(
        self,
        app: ASGIApp,
        requests_per_minute: int = 60,
        enable_rate_limiting: bool = False,
    ):
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.enable_rate_limiting = enable_rate_limiting
        self.request_counts: Dict[str, Dict[str, Any]] = {}

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        # Skip rate limiting if disabled
        if not self.enable_rate_limiting:
            return await call_next(request)

        # Get client IP
        client_ip = request.client.host if request.client else "unknown"
        current_time = time.time()

        # Initialize or update request count for this client
        if client_ip not in self.request_counts:
            self.request_counts[client_ip] = {
                "count": 0,
                "reset_time": current_time + 60,  # Reset after 1 minute
            }
        elif current_time > self.request_counts[client_ip]["reset_time"]:
            # Reset counter if time window has passed
            self.request_counts[client_ip] = {
                "count": 0,
                "reset_time": current_time + 60,
            }

        # Increment request count
        self.request_counts[client_ip]["count"] += 1

        # Check if rate limit exceeded
        if self.request_counts[client_ip]["count"] > self.requests_per_minute:
            request_id = getattr(request.state, "request_id", str(uuid.uuid4()))

            # Log rate limit exceeded
            logger.warning(f"Rate limit exceeded for client {client_ip}")
            request_logger.log_error(
                request_id=request_id,
                status_code=429,
                error="Rate limit exceeded",
                detailed=False,
            )

            # Create rate limit response
            headers = {
                "X-RateLimit-Limit": str(self.requests_per_minute),
                "X-RateLimit-Remaining": "0",
                "X-RateLimit-Reset": str(
                    int(self.request_counts[client_ip]["reset_time"])
                ),
                "X-Request-ID": request_id,
            }

            content = {
                "error": {
                    "type": "rate_limit_exceeded",
                    "message": "Rate limit exceeded. Please try again later.",
                    "status_code": 429,
                    "request_id": request_id,
                }
            }

            content_bytes = json.dumps(content).encode("utf-8")

            return Response(
                content=content_bytes,
                status_code=429,
                headers=headers,
                media_type="application/json",
            )

        # Add rate limit headers
        response = await call_next(request)
        remaining = self.requests_per_minute - self.request_counts[client_ip]["count"]
        response.headers["X-RateLimit-Limit"] = str(self.requests_per_minute)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        response.headers["X-RateLimit-Reset"] = str(
            int(self.request_counts[client_ip]["reset_time"])
        )

        return response
