from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

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

    default_dist = Path(__file__).resolve().parents[3] / "web" / "dist"
    web_dist = Path(os.getenv("WEB_DIST", str(default_dist)))
    serve_web = os.getenv("SERVE_WEB", "0") == "1" and web_dist.is_dir()

    if not serve_web:

        @app.get("/", include_in_schema=False)
        def root() -> dict:
            return {
                "service": "Research Team API",
                "docs": "/docs",
                "health": "/api/health",
            }

    app.include_router(router)

    if serve_web:
        app.mount("/", StaticFiles(directory=str(web_dist), html=True), name="web")

    return app


app = create_app()
