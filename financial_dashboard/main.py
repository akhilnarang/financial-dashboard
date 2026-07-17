"""Application factory for financial-dashboard."""

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import Depends, FastAPI
from fastapi.staticfiles import StaticFiles

from financial_dashboard.api import router as api_router
from financial_dashboard.config import settings
from financial_dashboard.core.deps import verify_credentials
from financial_dashboard.db import async_session, engine, init_db
from financial_dashboard.services.extensions import bootstrap_extensions
from financial_dashboard.services.fetch import FetchService
from financial_dashboard.services.settings import (
    assert_master_key_or_no_secrets,
    start_services,
    stop_services,
)
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
    # Register builtin extensions before init_db() runs load_all_settings(), so
    # contributed SettingDef entries are present when the settings cache fills.
    app.state.extension_manager = bootstrap_extensions()

    logger.info("Initializing database...")
    await init_db()
    logger.info("Database ready")

    # Refuse to boot with no master key if encrypted data already exists,
    # otherwise an ephemeral key would silently orphan those secrets.
    async with async_session() as session:
        await assert_master_key_or_no_secrets(session)

    await start_services()
    fetch_service = FetchService(extension_manager=app.state.extension_manager)
    app.state.fetch_service = fetch_service
    # Start extension runtimes now that the DB + settings are ready. This is
    # inert for Paisa (startup never starts Paisa or enables auto-sync); it only
    # marks runtimes ready to receive after-fetch-cycle hooks from the poll loop.
    await app.state.extension_manager.startup_all()
    await fetch_service.start_poll_loop()

    if not settings.auth_enabled:
        logger.warning(
            "No AUTH_USERNAME/AUTH_PASSWORD set — running without authentication. "
            "Only run on a trusted network or behind a reverse proxy with auth."
        )

    yield

    await fetch_service.stop_poll_loop()
    await app.state.extension_manager.shutdown_all()
    await stop_services()
    await engine.dispose()


def create_app() -> FastAPI:
    app = FastAPI(
        title="Ledger",
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
