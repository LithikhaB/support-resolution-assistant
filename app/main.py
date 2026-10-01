from fastapi import FastAPI

from app.api.routes import router
from app.config.settings import get_settings
from app.monitoring.logging import configure_logging

settings = get_settings()
configure_logging(settings.log_level)

app = FastAPI(title=settings.app_name, version="0.1.0")
app.include_router(router)