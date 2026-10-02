import psycopg
from fastapi import APIRouter
from fastapi.responses import JSONResponse

from app.config.settings import get_settings
from app.database.connection import get_connection

router = APIRouter(prefix="/api/v1")


@router.get("/health")
def health() -> dict[str, str]:
    """Report process liveness independently of database availability."""
    return {"status": "ok", "service": get_settings().app_name}


@router.get("/ready")
def readiness():
    """Check database schema availability; retrieval readiness is checked separately."""
    try:
        with get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("""
                    SELECT EXISTS(SELECT 1 FROM pg_extension WHERE extname = 'vector'),
                           to_regclass('public.documents') IS NOT NULL,
                           to_regclass('public.chunks') IS NOT NULL,
                           (SELECT count(*) = 4 FROM information_schema.columns
                            WHERE table_schema='public' AND table_name='documents'
                              AND column_name IN ('response', 'outcome_status', 'ticket_type', 'priority'))
                """)
                checks = cursor.fetchone()
        if not checks or not all(checks):
            return JSONResponse(
                status_code=503, content={"status": "not_ready", "reason": "database_schema"}
            )
    except psycopg.Error:
        return JSONResponse(
            status_code=503, content={"status": "not_ready", "reason": "database_unavailable"}
        )
    return {"status": "ready", "scope": "database_schema"}
