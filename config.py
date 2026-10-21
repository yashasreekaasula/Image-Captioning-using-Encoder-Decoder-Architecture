"""
Central configuration for the Image Captioning Studio backend.
All tuneable constants and model identifiers live here.
"""

from pydantic_settings import BaseSettings
from typing import Literal


class Settings(BaseSettings):
    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False

    # Allowed image types
    allowed_extensions: list[str] = ["jpg", "jpeg", "png"]
    allowed_mime_types: list[str] = ["image/jpeg", "image/png"]
    max_upload_bytes: int = 10 * 1024 * 1024  # 10 MB

    # Model identifiers
    vit_model_id: str = "google/vit-base-patch16-224"
    clip_model_id: str = "openai/clip-vit-base-patch32"
    blip_model_id: str = "Salesforce/blip-image-captioning-base"
    yolo_model_id: str = "yolov8n.pt"

    # LSTM decoder vocab (toy – replace with trained vocab in production)
    lstm_vocab_size: int = 10000
    lstm_embed_dim: int = 256
    lstm_hidden_dim: int = 512
    lstm_encoder_dim: int = 2048  # ResNet50 output

    # Transformer decoder
    transformer_vocab_size: int = 10000
    transformer_d_model: int = 512
    transformer_nhead: int = 8
    transformer_num_layers: int = 4
    transformer_max_seq_len: int = 50

    # Device
    device: str = "cpu"  # Change to "cuda" when GPU is available

    class Config:
        env_file = ".env"


settings = Settings()