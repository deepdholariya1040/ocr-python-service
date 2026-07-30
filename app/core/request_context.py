"""
Assigns a request id to every incoming request (reusing an inbound
X-Request-Id header if the caller already set one), stashes it on
`request.state`, and echoes it back on the response so a single scan can be
traced end-to-end across the Node backend's logs and this service's logs.
"""

from __future__ import annotations

import uuid

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

REQUEST_ID_HEADER = "X-Request-Id"


class RequestIdMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        request_id = request.headers.get(REQUEST_ID_HEADER) or str(uuid.uuid4())
        request.state.request_id = request_id

        response = await call_next(request)
        response.headers[REQUEST_ID_HEADER] = request_id
        return response
