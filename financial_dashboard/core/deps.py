"""FastAPI dependencies for financial-dashboard."""

from typing import Annotated, Optional

from fastapi import Depends, Request
from fastapi.security import HTTPBasicCredentials
from sqlalchemy.ext.asyncio import AsyncSession

from financial_dashboard.core.security import check_credentials, http_basic
from financial_dashboard.db import async_session

MCP_PATH = "/api/mcp"


def verify_credentials(
    request: Request,
    credentials: Optional[HTTPBasicCredentials] = Depends(http_basic),
) -> None:
    # The MCP endpoint checks its own bearer token. MCP clients send no Basic auth.
    if request.url.path == MCP_PATH:
        return
    check_credentials(credentials, request)


async def get_session():
    async with async_session() as session:
        yield session


AsyncSessionDep = Annotated[AsyncSession, Depends(get_session)]
