from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from ..config import load_settings
from .routes import router
from .sessions import SessionManager


def create_app(settings=None, manager: SessionManager | None = None) -> FastAPI:
    settings = settings or load_settings()
    app = FastAPI(title="Research Team API", version="0.1.0")

    app.state.settings = settings
    app.state.sessions = manager or SessionManager(settings)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/", include_in_schema=False)
    def root() -> dict:
        return {
            "service": "Research Team API",
            "docs": "/docs",
            "health": "/api/health",
        }

    app.include_router(router)
    return app


app = create_app()
