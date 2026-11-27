"""The MCP endpoint: one stateless JSON-RPC message per POST."""

import json
import secrets

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from pydantic import JsonValue

from financial_dashboard.config import settings
from financial_dashboard.core.deps import AsyncSessionDep
from financial_dashboard.services.mcp import (
    PARSE_ERROR,
    PROTOCOL_VERSIONS,
    JsonRpcResponse,
    error_response,
    handle_message,
)

router = APIRouter()


def verify_mcp_request(request: Request) -> None:
    """Accept only a non-browser client with the MCP bearer token.

    A browser always sends Origin on a cross-site POST, and MCP clients do not,
    so any Origin is refused. That blocks DNS rebinding and cross-site posts.

    Args:
        request: The incoming request.

    Raises:
        HTTPException: 404 when no token is set, 403 for a browser origin,
            401 for a missing or wrong token, 400 for an unknown protocol
            version.
    """
    token = settings.mcp_token.get_secret_value()
    if not token:
        raise HTTPException(status_code=404)
    if request.headers.get("origin") is not None:
        raise HTTPException(status_code=403, detail="browser origins are refused")
    scheme, _, given = request.headers.get("authorization", "").partition(" ")
    if scheme.lower() != "bearer" or not secrets.compare_digest(
        given.encode(), token.encode()
    ):
        raise HTTPException(
            status_code=401, headers={"WWW-Authenticate": 'Bearer realm="mcp"'}
        )
    version = request.headers.get("mcp-protocol-version")
    if version is not None and version not in PROTOCOL_VERSIONS:
        raise HTTPException(status_code=400, detail="unsupported MCP protocol version")


@router.post("/mcp", dependencies=[Depends(verify_mcp_request)])
async def mcp(request: Request, session: AsyncSessionDep) -> Response:
    """Answer one JSON-RPC message.

    A request gets a JSON reply. A notification or a client response gets 202
    with no body.
    """
    try:
        message = json.loads(await request.body())
    except ValueError:
        reply = error_response(None, PARSE_ERROR, "the body is not JSON")
        return JSONResponse(_reply_body(reply), 400)
    reply = await handle_message(session, message)
    if reply is None:
        return Response(status_code=202)
    return JSONResponse(_reply_body(reply))


@router.get("/mcp", dependencies=[Depends(verify_mcp_request)])
async def mcp_stream() -> Response:
    """Refuse the event stream. This server answers each POST and sends nothing else."""
    return Response(status_code=405, headers={"Allow": "POST"})


@router.delete("/mcp", dependencies=[Depends(verify_mcp_request)])
async def mcp_end_session() -> Response:
    """Refuse a session end. The server keeps no session."""
    return Response(status_code=405, headers={"Allow": "POST"})


def _reply_body(reply: JsonRpcResponse) -> dict[str, JsonValue]:
    """Serialize a reply. JSON-RPC requires the id, even when it is null."""
    return {"id": None, **reply.model_dump(mode="json", exclude_none=True)}
