

from __future__ import annotations

import torch
from loguru import logger
from typing import Optional


def build_qlora_blip(model_id: str = "Salesforce/blip-image-captioning-base"):
    """
    Returns a BLIP model wrapped with QLoRA adapters.

    Requires: bitsandbytes, peft, accelerate
    """
    try:
        from transformers import (
            BlipForConditionalGeneration,
            BlipProcessor,
            BitsAndBytesConfig,
        )
        from peft import LoraConfig, get_peft_model, TaskType
    except ImportError as exc:
        raise ImportError(
            "Install bitsandbytes and peft: pip install bitsandbytes peft accelerate"
        ) from exc

    # ------------------------------------------------------------------ #
    # 1. 4-bit quantisation config                                         #
    # ------------------------------------------------------------------ #
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",          # NormalFloat4 – best quality
        bnb_4bit_use_double_quant=True,      # Second quantisation of constants
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    logger.info(f"Loading {model_id} in 4-bit for QLoRA …")
    model = BlipForConditionalGeneration.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
    )

    lora_config = LoraConfig(
        r=16,                            # Rank of the low-rank matrices
        lora_alpha=32,                   # Scaling α/r
        target_modules=["query", "value"],  # Target BLIP text decoder attn
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM,
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    processor = BlipProcessor.from_pretrained(model_id)
    return model, processor


class QLoRATrainer:
    """
    Thin wrapper around a HuggingFace Trainer for BLIP QLoRA fine-tuning.
    Replace the dataset loading with your actual COCO / custom dataset.
    """

    def __init__(self, model_id: str = "Salesforce/blip-image-captioning-base"):
        self.model_id = model_id
        self.model = None
        self.processor = None

    def setup(self):
        """Prepare model + processor."""
        self.model, self.processor = build_qlora_blip(self.model_id)
        logger.info("QLoRA BLIP model ready for training.")

    def train(
        self,
        train_dataset,
        output_dir: str = "blip-qlora-output",
        epochs: int = 3,
        batch_size: int = 4,
        lr: float = 2e-4,
    ):
        """
        Fine-tune using HuggingFace Trainer.

        Args:
            train_dataset: HuggingFace Dataset with 'image' and 'caption' columns.
            output_dir:    Where to save LoRA adapter weights.
        """
        try:
            from transformers import TrainingArguments, Trainer
        except ImportError as exc:
            raise ImportError("Install transformers>=4.35") from exc

        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=4,
            learning_rate=lr,
            fp16=False,
            bf16=True,
            logging_steps=50,
            save_steps=500,
            save_total_limit=2,
            report_to="none",
            remove_unused_columns=False,
        )

        def collate_fn(batch):
            images = [item["image"] for item in batch]
            captions = [item["caption"] for item in batch]
            encoding = self.processor(
                images=images,
                text=captions,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=64,
            )
            # BLIP uses input_ids as labels for conditional generation
            encoding["labels"] = encoding["input_ids"].clone()
            return encoding

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=collate_fn,
        )

        logger.info("Starting QLoRA fine-tuning …")
        trainer.train()
        logger.info(f"Training complete. Saving to {output_dir}")

        # Save only LoRA adapter weights (very small ~MB range)
        self.model.save_pretrained(output_dir)
        self.processor.save_pretrained(output_dir)
        logger.info("Adapter weights saved.")