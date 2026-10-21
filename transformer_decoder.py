
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List
from loguru import logger

from .base import BaseDecoder
from backend.config import settings


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class _TransformerDecoderModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        encoder_dim: int,
        max_seq_len: int = 50,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        # Project encoder features into d_model space
        self.encoder_proj = nn.Linear(encoder_dim, d_model)

        # Token embedding + positional encoding
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len=max_seq_len + 1, dropout=dropout)

        # Standard PyTorch transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,  # (B, S, D) convention
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=num_layers
        )

        self.fc_out = nn.Linear(d_model, vocab_size)
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc_out.weight)
        nn.init.zeros_(self.fc_out.bias)

    def _make_causal_mask(self, sz: int, device: torch.device) -> torch.Tensor:
        """Upper-triangular mask so position i cannot attend to j > i."""
        mask = torch.triu(torch.ones(sz, sz, device=device), diagonal=1).bool()
        return mask

    def forward(
        self,
        encoder_out: torch.Tensor,
        max_len: int = 20,
        start_token: int = 1,
        end_token: int = 2,
    ) -> Tuple[List[int], List[float]]:
        """
        Autoregressive greedy decoding.

        Args:
            encoder_out: (1, encoder_dim) or (1, S, encoder_dim)
        """
        device = encoder_out.device

        # Ensure memory is (1, S, d_model)
        if encoder_out.dim() == 2:
            memory = self.encoder_proj(encoder_out.unsqueeze(1))  # (1,1,d_model)
        else:
            memory = self.encoder_proj(encoder_out)               # (1,S,d_model)

        # Start with <start> token
        generated = [start_token]
        confidences: List[float] = []

        for step in range(max_len):
            tgt_ids = torch.LongTensor([generated]).to(device)  # (1, step+1)
            tgt_embed = self.pos_enc(
                self.token_embed(tgt_ids) * math.sqrt(self.d_model)
            )  # (1, step+1, d_model)

            causal_mask = self._make_causal_mask(len(generated), device)
            out = self.transformer_decoder(
                tgt=tgt_embed,
                memory=memory,
                tgt_mask=causal_mask,
            )  # (1, step+1, d_model)

            logits = self.fc_out(out[:, -1, :])   # (1, vocab_size)
            probs = F.softmax(logits, dim=-1)
            next_token = probs.argmax(dim=-1).item()
            conf = probs.max(dim=-1).values.item()

            if next_token == end_token:
                break

            generated.append(next_token)
            confidences.append(conf)

        token_ids = generated[1:]  # strip <start>
        return token_ids, confidences


class TransformerCaptionDecoder(BaseDecoder):
    """
    Transformer Decoder caption generator.
    Returns placeholder until real weights are loaded.
    """

    _TOY_VOCAB = {
        0: "<pad>", 1: "<start>", 2: "<end>",
        3: "a", 4: "photograph", 5: "depicting", 6: "various",
        7: "objects", 8: "and", 9: "elements",
    }

    def __init__(self, encoder_dim: int = 2048):
        logger.info("Initialising Transformer decoder …")
        self._encoder_dim = encoder_dim
        self._model = _TransformerDecoderModel(
            vocab_size=settings.transformer_vocab_size,
            d_model=settings.transformer_d_model,
            nhead=settings.transformer_nhead,
            num_layers=settings.transformer_num_layers,
            encoder_dim=encoder_dim,
            max_seq_len=settings.transformer_max_seq_len,
        )
        self._model.eval()
        self._weights_loaded = False
        logger.info("Transformer decoder ready (random weights).")

    def load_weights(self, path: str):
        state = torch.load(path, map_location="cpu")
        self._model.load_state_dict(state)
        self._weights_loaded = True
        logger.info(f"Transformer weights loaded from {path}")

    def decode(self, features: torch.Tensor) -> Tuple[str, float]:
        if not self._weights_loaded:
            return (
                "a detailed visual scene captured in the photograph "
                "[train Transformer for real captions]",
                0.38,
            )

        with torch.no_grad():
            token_ids, confs = self._model(features)

        words = [self._TOY_VOCAB.get(t, f"<{t}>") for t in token_ids]
        caption = " ".join(words).strip()
        confidence = float(sum(confs) / len(confs)) if confs else 0.0
        return caption, round(confidence, 4)