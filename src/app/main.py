from __future__ import annotations

import platform
import time
from typing import Any, Dict

from fastapi import FastAPI
from app.api.query import router as query_router
from app.api.evaluate import router as eval_router

from app import __version__
from app.config.settings import get_settings
from app.logging.json_logger import configure_json_logging

app = FastAPI(title="Agentic RAG Benchmarking POC", version=__version__)
app.include_router(query_router)
app.include_router(eval_router)


@app.on_event("startup")
def _startup() -> None:
    settings = get_settings()
    configure_json_logging(settings.log_level)


@app.get("/health")
def health() -> Dict[str, Any]:
    settings = get_settings()
    # GPU status (Mac): unavailable by default for this POC
    gpu_status = {"available": False, "details": {"device": None}}
    model_status = {
        "providers": {
            "openai": bool(settings.openai_api_key),
            "gemini": bool(settings.gemini_api_key),
        },
        "loaded": False,
    }
    vectordb_status = {
        "provider": "qdrant",
        "configured": bool(settings.qdrant_url and settings.qdrant_api_key),
        "collection": settings.qdrant_collection,
    }
    return {
        "status": "healthy",
        "timestamp": int(time.time()),
        "version": __version__,
        "system": {"python": platform.python_version(), "platform": platform.platform()},
        "model": model_status,
        "gpu": gpu_status,
        "vectordb": vectordb_status,
        "last_successful_prediction_at": None,
    }


