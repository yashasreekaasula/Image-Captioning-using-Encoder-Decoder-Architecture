"""
LSTM + Bahdanau Attention decoder implemented from scratch.

Architecture:
  - Encoder features projected to `hidden_dim` as initial LSTM state.
  - At each step the attention mechanism computes a context vector
    as a weighted sum of the projected encoder "spatial" tokens.
  - LSTM input = concat(word_embed, context_vector).
  - Linear head maps LSTM output → vocab logits.

NOTE: Without a trained vocabulary / weights this module returns a
      placeholder caption. Plug in real weights via `load_weights()`.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List
from loguru import logger

from .base import BaseDecoder
from backend.config import settings


# ---------------------------------------------------------------------------
# Bahdanau Attention
# ---------------------------------------------------------------------------

class BahdanauAttention(nn.Module):
    """
    Additive (Bahdanau) attention.

    score(h_t, h_s) = v^T · tanh(W_h · h_t + W_s · h_s)
    """

    def __init__(self, hidden_dim: int, encoder_dim: int, attention_dim: int = 256):
        super().__init__()
        self.W_h = nn.Linear(hidden_dim, attention_dim, bias=False)
        self.W_s = nn.Linear(encoder_dim, attention_dim, bias=False)
        self.v = nn.Linear(attention_dim, 1, bias=False)

    def forward(
        self, hidden: torch.Tensor, encoder_out: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            hidden:      (B, hidden_dim) – current decoder hidden state
            encoder_out: (B, S, encoder_dim) – S spatial feature tokens

        Returns:
            context:  (B, encoder_dim) weighted sum
            weights:  (B, S) attention weights
        """
        # (B, 1, attention_dim) + (B, S, attention_dim) → (B, S, attention_dim)
        energy = self.v(
            torch.tanh(self.W_h(hidden).unsqueeze(1) + self.W_s(encoder_out))
        ).squeeze(2)  # (B, S)

        weights = F.softmax(energy, dim=1)              # (B, S)
        context = (weights.unsqueeze(2) * encoder_out).sum(1)  # (B, encoder_dim)
        return context, weights


# ---------------------------------------------------------------------------
# LSTM Decoder
# ---------------------------------------------------------------------------

class _LSTMDecoderModel(nn.Module):
    """
    The actual PyTorch module – separated so we can load/save weights cleanly.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_dim: int,
        encoder_dim: int,
        attention_dim: int = 256,
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.attention = BahdanauAttention(hidden_dim, encoder_dim, attention_dim)

        # Encoder → initial hidden & cell states
        self.init_h = nn.Linear(encoder_dim, hidden_dim)
        self.init_c = nn.Linear(encoder_dim, hidden_dim)

        self.lstm = nn.LSTMCell(embed_dim + encoder_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(0.3)

    def forward(
        self,
        encoder_out: torch.Tensor,
        max_len: int = 20,
        start_token: int = 1,
        end_token: int = 2,
    ) -> Tuple[List[int], List[float]]:
        """
        Greedy decode.

        Args:
            encoder_out: (1, encoder_dim) – global feature (will be treated
                         as a single "spatial token" if 2-D, or (1,S,D) if 3-D)

        Returns:
            token_ids, step_confidences
        """
        B = 1
        device = encoder_out.device

        # If 2-D, expand to (1, 1, D) so attention has something to attend over
        if encoder_out.dim() == 2:
            enc = encoder_out.unsqueeze(1)  # (1, 1, D)
        else:
            enc = encoder_out  # (1, S, D)

        # Initial hidden / cell from mean of encoder tokens
        mean_enc = enc.mean(1)  # (1, D)
        h = torch.tanh(self.init_h(mean_enc))
        c = torch.tanh(self.init_c(mean_enc))

        token = torch.LongTensor([start_token]).to(device)
        token_ids: List[int] = []
        confidences: List[float] = []

        for _ in range(max_len):
            embed = self.embed(token)                 # (1, embed_dim)
            context, _ = self.attention(h, enc)       # (1, encoder_dim)
            lstm_input = torch.cat([embed, context], dim=1)
            h, c = self.lstm(lstm_input, (h, c))
            logits = self.fc_out(self.dropout(h))     # (1, vocab_size)
            probs = F.softmax(logits, dim=1)
            token = probs.argmax(dim=1)
            conf = probs.max(dim=1).values.item()

            if token.item() == end_token:
                break

            token_ids.append(token.item())
            confidences.append(conf)

        return token_ids, confidences


# ---------------------------------------------------------------------------
# Public wrapper satisfying BaseDecoder
# ---------------------------------------------------------------------------

class LSTMAttentionDecoder(BaseDecoder):
    """
    LSTM + Bahdanau Attention caption decoder.
    In production, call `load_weights(path)` after initialising.
    Without weights, returns a placeholder caption to show the pipeline works.
    """

    # Minimal toy vocabulary for demonstration
    _TOY_VOCAB = {
        0: "<pad>", 1: "<start>", 2: "<end>",
        3: "a", 4: "an", 5: "the", 6: "image", 7: "shows",
        8: "of", 9: "with", 10: "in", 11: "on", 12: "and",
    }

    def __init__(self, encoder_dim: int = 2048):
        logger.info("Initialising LSTM+Attention decoder …")
        self._encoder_dim = encoder_dim
        self._model = _LSTMDecoderModel(
            vocab_size=settings.lstm_vocab_size,
            embed_dim=settings.lstm_embed_dim,
            hidden_dim=settings.lstm_hidden_dim,
            encoder_dim=encoder_dim,
        )
        self._model.eval()
        self._weights_loaded = False
        logger.info("LSTM decoder ready (random weights – load trained weights for real captions).")

    def load_weights(self, path: str):
        """Load trained model weights from a .pt checkpoint."""
        state = torch.load(path, map_location="cpu")
        self._model.load_state_dict(state)
        self._weights_loaded = True
        logger.info(f"LSTM weights loaded from {path}")

    def decode(self, features: torch.Tensor) -> Tuple[str, float]:
        """
        Generate a caption.  Without trained weights returns placeholder text.
        """
        if not self._weights_loaded:
            # Return human-readable placeholder so UI works end-to-end
            return (
                "a scene captured in the image [train LSTM for real captions]",
                0.42,
            )

        with torch.no_grad():
            token_ids, confs = self._model(features)

        words = [self._TOY_VOCAB.get(t, f"<{t}>") for t in token_ids]
        caption = " ".join(words).strip()
        confidence = float(sum(confs) / len(confs)) if confs else 0.0
        return caption, round(confidence, 4)