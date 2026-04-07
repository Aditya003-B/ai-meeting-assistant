"""FastAPI application entry point."""

from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import meetings, query, transcribe
from app.core.config import settings
from app.core.database import engine
from app.utils.logger import get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> Any:
    """Application lifespan context manager for startup and shutdown."""
    logger.info("Starting AI Meeting Assistant API")
    logger.info(f"Environment: {settings.environment}")
    logger.info(f"LLM Model: {settings.llm_model}")
    logger.info(f"Embedding Model: {settings.embedding_model}")
    
    yield
    
    logger.info("Shutting down AI Meeting Assistant API")
    await engine.dispose()


app = FastAPI(
    title="AI Meeting Assistant",
    description="Agentic meeting assistant with transcription, extraction, and RAG capabilities",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "https://ai-meeting-assistant-ui.vercel.app",
        "https://*.vercel.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(transcribe.router)
app.include_router(meetings.router)
app.include_router(query.router)


@app.get(
    "/",
    tags=["health"],
    summary="Health check",
    description="Check if the API is running",
)
async def health_check() -> dict[str, str]:
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "AI Meeting Assistant",
        "version": "0.1.0",
    }


@app.get(
    "/health",
    tags=["health"],
    summary="Detailed health check",
    description="Get detailed health information",
)
async def detailed_health() -> dict[str, Any]:
    """Detailed health check endpoint."""
    return {
        "status": "healthy",
        "service": "AI Meeting Assistant",
        "version": "0.1.0",
        "environment": settings.environment,
        "llm_model": settings.llm_model,
        "embedding_model": settings.embedding_model,
    }
