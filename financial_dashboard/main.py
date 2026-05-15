"""Application factory for financial-dashboard."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import Depends, FastAPI
from fastapi.staticfiles import StaticFiles

from financial_dashboard.api import router as api_router
from financial_dashboard.config import settings
from financial_dashboard.core.deps import verify_credentials
from financial_dashboard.db import engine, init_db
from financial_dashboard.services.fetch import FetchService
from financial_dashboard.services.settings import start_services, stop_services
from financial_dashboard.web import get_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("telegram").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing database...")
    await init_db()
    logger.info("Database ready")

    await start_services()
    fetch_service = FetchService()
    app.state.fetch_service = fetch_service
    await fetch_service.start_poll_loop()

    if not settings.auth_enabled:
        logger.warning(
            "No AUTH_USERNAME/AUTH_PASSWORD set — running without authentication. "
            "Only run on a trusted network or behind a reverse proxy with auth."
        )

    yield

    await fetch_service.stop_poll_loop()
    await stop_services()
    await engine.dispose()


def create_app() -> FastAPI:
    app = FastAPI(
        title="Email Fetcher",
        lifespan=lifespan,
        dependencies=[Depends(verify_credentials)],
    )
    app.mount(
        "/static",
        StaticFiles(directory=Path(__file__).resolve().parent / "static"),
        name="static",
    )
    app.include_router(api_router)
    app.include_router(get_router())
    return app
