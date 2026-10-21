"""
Training script for LSTM + Attention image captioning model
(Final corrected, Windows-safe, CPU-friendly version)

Dataset structure:
data/
 ├── images/
 │     ├── *.jpg
 └── captions.json   [{"image": "...jpg", "caption": "..."}]

Usage examples:
python -m training.train_lstm --epochs 1
python -m training.train_lstm --epochs 1 --batch_size 8 --max_samples 1000
"""

import argparse
import json
import os
import time
from collections import Counter
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from loguru import logger

from backend.models.encoders.resnet_encoder import ResNetEncoder
from backend.models.decoders.lstm_decoder import _LSTMDecoderModel
from backend.config import settings


# ============================================================
# Vocabulary
# ============================================================

class Vocabulary:
    PAD, START, END, UNK = 0, 1, 2, 3
    SPECIAL = ["<pad>", "<start>", "<end>", "<unk>"]

    def __init__(self, freq_threshold=5):
        self.freq_threshold = freq_threshold
        self.itos = {i: tok for i, tok in enumerate(self.SPECIAL)}
        self.stoi = {tok: i for i, tok in self.itos.items()}

    def build(self, captions: List[str]):
        counter = Counter()
        for cap in captions:
            counter.update(cap.lower().split())

        for word, freq in counter.items():
            if freq >= self.freq_threshold and word not in self.stoi:
                idx = len(self.itos)
                self.itos[idx] = word
                self.stoi[word] = idx

        logger.info(f"Vocabulary size: {len(self.itos)}")

    def encode(self, caption: str):
        tokens = [self.START]
        tokens += [self.stoi.get(w, self.UNK) for w in caption.lower().split()]
        tokens.append(self.END)
        return tokens

    def __len__(self):
        return len(self.itos)


# ============================================================
# Dataset
# ============================================================

class CaptionDataset(Dataset):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    def __init__(self, data_dir: str, vocab: Vocabulary, max_len=40, max_samples=None):
        self.data_dir = Path(data_dir)
        self.vocab = vocab
        self.max_len = max_len

        with open(self.data_dir / "captions.json", "r", encoding="utf-8") as f:
            raw = json.load(f)

        if max_samples is not None:
            raw = raw[:max_samples]

        self.samples: List[Tuple[str, str]] = [
            (r["image"], r["caption"]) for r in raw
        ]

        logger.info(f"Dataset size: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fname, caption = self.samples[idx]
        img = Image.open(self.data_dir / "images" / fname).convert("RGB")
        img = self.transform(img)

        tokens = self.vocab.encode(caption)
        tokens = tokens[: self.max_len]

        return img, torch.tensor(tokens, dtype=torch.long)


def collate_fn(batch):
    images, captions = zip(*batch)
    images = torch.stack(images)

    lengths = [len(c) for c in captions]
    max_len = max(lengths)

    padded = torch.zeros(len(captions), max_len, dtype=torch.long)
    for i, cap in enumerate(captions):
        padded[i, :len(cap)] = cap

    return images, padded, torch.tensor(lengths)


# ============================================================
# Training
# ============================================================

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training on {device}")

    # Load captions for vocab
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
        max_len=40,
        max_samples=args.max_samples
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,       # 🔑 Windows-safe
        pin_memory=False
    )

    # Encoder (frozen)
    encoder = ResNetEncoder()
    encoder._backbone.eval()
    encoder._backbone.to(device)

    # Decoder
    decoder = _LSTMDecoderModel(
        vocab_size=len(vocab),
        embed_dim=settings.lstm_embed_dim,
        hidden_dim=settings.lstm_hidden_dim,
        encoder_dim=2048,
    ).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=Vocabulary.PAD)
    optimizer = optim.Adam(decoder.parameters(), lr=args.lr)

    best_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        decoder.train()
        total_loss = 0.0
        start = time.time()

        for i, (images, captions, lengths) in enumerate(loader):
            print(f"Batch {i}")

            images = images.to(device)
            captions = captions.to(device)

            with torch.no_grad():
                feats = encoder._backbone(images)
                feats = feats.view(images.size(0), -1)

            enc_out = feats.unsqueeze(1)
            mean_enc = enc_out.mean(1)

            h = torch.tanh(decoder.init_h(mean_enc))
            c = torch.tanh(decoder.init_c(mean_enc))

            loss = 0.0
            max_len = captions.size(1)

            for t in range(max_len - 1):
                embed = decoder.embed(captions[:, t])
                context, _ = decoder.attention(h, enc_out)
                lstm_in = torch.cat([embed, context], dim=1)
                h, c = decoder.lstm(lstm_in, (h, c))
                logits = decoder.fc_out(h)
                loss += criterion(logits, captions[:, t + 1])

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), 5.0)
            optimizer.step()

            total_loss += loss.item() / (max_len - 1)

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
            logger.info(f"✓ Saved best model to {args.output}")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/")
    parser.add_argument("--output", default="checkpoints/lstm.pt")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=4)     # CPU safe
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--max_samples", type=int, default=None) # 🔥 NEW

    train(parser.parse_args())