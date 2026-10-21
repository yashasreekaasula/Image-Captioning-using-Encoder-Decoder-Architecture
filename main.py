"""
FastAPI application entry point for the Image Captioning Studio.
"""

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from backend.routes import router
from backend.config import settings

logger.add(
    "logs/backend.log",
    rotation="10 MB",
    retention="7 days",
    level="INFO",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
)

app = FastAPI(
    title="Image Captioning Studio API",
    description="Modular image captioning with YOLO, ResNet/ViT/CLIP encoders "
                "and LSTM/Transformer/BLIP decoders.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1")


@app.on_event("startup")
async def startup_event():
    logger.info("Image Captioning Studio backend starting …")


@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Image Captioning Studio backend shutting down.")


if __name__ == "__main__":
    uvicorn.run(
        "backend.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
    )