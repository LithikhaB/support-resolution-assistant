from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api.resolution import router as resolution_router
from app.api.retrieval import router as retrieval_router
from app.api.routes import router
from app.api.understanding import router as understanding_router
from app.config.settings import get_settings
from app.database.connection import close_pools
from app.monitoring.logging import configure_logging
from app.monitoring.metrics import measure_request, prometheus_metrics
from app.monitoring.metrics import router as metrics_router

settings = get_settings()
configure_logging(settings.log_level)


@asynccontextmanager
async def lifespan(app):
    yield
    close_pools()


app = FastAPI(title=settings.app_name, version="0.1.0", lifespan=lifespan)
app.middleware("http")(measure_request)
app.include_router(metrics_router)
app.include_router(router)
app.include_router(retrieval_router)
app.include_router(understanding_router)
app.include_router(resolution_router)
app.add_api_route("/metrics", prometheus_metrics, include_in_schema=False)

WEB_DIR = Path(__file__).parent / "web"
app.mount("/assets", StaticFiles(directory=WEB_DIR), name="assets")


@app.get("/", include_in_schema=False)
def home():
    """Open the complaint form without loading models."""
    return FileResponse(WEB_DIR / "index.html")


@app.get("/workspace", include_in_schema=False)
def workspace():
    """Serve the local agent workspace without loading any machine-learning models."""
    return FileResponse(WEB_DIR / "index.html")
