"""API router aggregation."""

from fastapi import APIRouter

from .polling import router as polling_router
from .sms import router as sms_router
from .sources import router as sources_router
from .transactions import router as transactions_router

router = APIRouter(prefix="/api")
router.include_router(polling_router)
router.include_router(transactions_router)
router.include_router(sources_router)
router.include_router(sms_router)
