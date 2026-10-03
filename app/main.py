from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api.resolution import router as resolution_router
from app.api.retrieval import router as retrieval_router
from app.api.routes import router
from app.api.understanding import router as understanding_router
from app.api.workflow import router as workflow_router
from app.config.settings import get_settings
from app.monitoring.logging import configure_logging
from app.monitoring.metrics import measure_request
from app.monitoring.metrics import router as metrics_router

settings = get_settings()
configure_logging(settings.log_level)

app = FastAPI(title=settings.app_name, version="0.1.0")
app.middleware("http")(measure_request)
app.include_router(metrics_router)
app.include_router(router)
app.include_router(retrieval_router)
app.include_router(understanding_router)
app.include_router(resolution_router)
app.include_router(workflow_router)

WEB_DIR = Path(__file__).parent / "web"
app.mount("/assets", StaticFiles(directory=WEB_DIR), name="assets")


@app.get("/", include_in_schema=False)
def home():
    """Serve the product introduction without loading models."""
    return FileResponse(WEB_DIR / "home.html")


@app.get("/workspace", include_in_schema=False)
def workspace():
    """Serve the local agent workspace without loading any machine-learning models."""
    return FileResponse(WEB_DIR / "index.html")
