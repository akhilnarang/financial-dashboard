from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from financial_dashboard.core.deps import get_session
from financial_dashboard.schemas.system import SystemInfoResponse
from financial_dashboard.services.system import get_system_info

router = APIRouter()


@router.get("/system/info")
async def system_info(
    session: AsyncSession = Depends(get_session),
) -> SystemInfoResponse:
    return await get_system_info(session)
