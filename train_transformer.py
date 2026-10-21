"""
Training script for Transformer Decoder image captioning model
(Final corrected, Windows-safe, CPU-friendly version)

Dataset structure:
data/
 ├── images/
 └── captions.json   [{"image": "...jpg", "caption": "..."}]

Usage:
python -m training.train_transformer --epochs 1 --batch_size 4 --max_samples 1000
"""

import argparse
import json
import os
import time
import math
from pathlib import Path
from typing import List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from loguru import logger

from training.train_lstm import Vocabulary, CaptionDataset, collate_fn
from backend.models.encoders.resnet_encoder import ResNetEncoder
from backend.models.decoders.transformer_decoder import _TransformerDecoderModel
from backend.config import settings


# ============================================================
# Training
# ============================================================

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training Transformer decoder on {device}")

    # Load captions
    with open(Path(args.data_dir) / "captions.json", "r", encoding="utf-8") as f:
        raw = json.load(f)

    if args.max_samples is not None:
        raw = raw[:args.max_samples]

    captions = [r["caption"] for r in raw]

    vocab = Vocabulary(freq_threshold=5)
    vocab.build(captions)

    dataset = CaptionDataset(
        args.data_dir,
        vocab,
        max_len=50,
        max_samples=args.max_samples,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,      # 🔑 Windows-safe
        pin_memory=False
    )

    # Encoder (frozen)
    encoder = ResNetEncoder()
    encoder._backbone.eval()
    encoder._backbone.to(device)

    # Decoder
    decoder = _TransformerDecoderModel(
        vocab_size=len(vocab),
        d_model=settings.transformer_d_model,
        nhead=settings.transformer_nhead,
        num_layers=settings.transformer_num_layers,
        encoder_dim=2048,
        max_seq_len=50,
    ).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=Vocabulary.PAD)
    optimizer = optim.AdamW(decoder.parameters(), lr=args.lr, weight_decay=1e-4)

    best_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        decoder.train()
        total_loss = 0.0
        start = time.time()

        for i, (images, captions, lengths) in enumerate(loader):
            print(f"Batch {i}")  # 🔥 visible progress

            images = images.to(device)
            captions = captions.to(device)

            # CNN encoding
            with torch.no_grad():
                feats = encoder._backbone(images)
                feats = feats.view(images.size(0), -1)  # (B, 2048)

            enc_out = feats.unsqueeze(1)  # (B, 1, 2048)
            memory = decoder.encoder_proj(enc_out)  # (B, 1, d_model)

            # Teacher forcing
            tgt_in = captions[:, :-1]
            tgt_out = captions[:, 1:]

            B, seq_len = tgt_in.size()
            tgt_embed = decoder.pos_enc(
                decoder.token_embed(tgt_in)
                * math.sqrt(settings.transformer_d_model)
            )

            causal_mask = decoder._make_causal_mask(seq_len, device)

            out = decoder.transformer_decoder(
                tgt=tgt_embed,
                memory=memory,
                tgt_mask=causal_mask,
            )

            logits = decoder.fc_out(out)
            logits = logits.view(-1, len(vocab))
            targets = tgt_out.reshape(-1)

            loss = criterion(logits, targets)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        elapsed = time.time() - start

        logger.info(
            f"Epoch [{epoch}/{args.epochs}] "
            f"Loss: {avg_loss:.4f} | Time: {elapsed:.1f}s"
        )

        if avg_loss < best_loss:
            best_loss = avg_loss
            os.makedirs(os.path.dirname(args.output), exist_ok=True)
            torch.save(decoder.state_dict(), args.output)
            logger.info(f"✓ Saved best transformer model to {args.output}")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/")
    parser.add_argument("--output", default="checkpoints/transformer.pt")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=4)      # CPU safe
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max_samples", type=int, default=None)  # 🔥 NEW

    train(parser.parse_args())