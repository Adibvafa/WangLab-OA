import os
from argparse import ArgumentParser
from typing import List, Dict, Optional

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

from peft import get_peft_model, LoraConfig, prepare_model_for_kbit_training
from transformers import (
    BitsAndBytesConfig,
    LlavaNextProcessor,
    LlavaNextForConditionalGeneration,
)

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger

from dataset import XRayReportDataset


def find_all_linear_names(model: torch.nn.Module) -> List[str]:
    """
    Find all linear layer names in the model for LoRA fine-tuning.

    Args:
        model (torch.nn.Module): The model to search.

    Returns:
        List[str]: A list of linear layer names.
    """
    lora_module_names = set()
    multimodal_keywords = ["multi_modal_projector", "vision_model"]

    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, torch.nn.Linear):
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if "lm_head" in lora_module_names:
        lora_module_names.remove("lm_head")

    return list(lora_module_names)


def load_processor(
    processor_dir: str, padding_side: str = "right"
) -> LlavaNextProcessor:
    """
    Load and configure the LlavaNextProcessor.

    Args:
        processor_dir (str): Directory containing the processor files.
        padding_side (str): Side to pad the inputs.

    Returns:
        LlavaNextProcessor: Configured processor.
    """
    processor = LlavaNextProcessor.from_pretrained(processor_dir)
    processor.tokenizer.padding_side = padding_side
    processor.tokenizer.pad_token = processor.tokenizer.eos_token
    return processor


def load_model(weights_dir: str, model_name: str) -> LlavaNextForConditionalGeneration:
    """
    Load, configure, and apply LoRA to the LlavaNextForConditionalGeneration model.

    Args:
        weights_dir (str): Directory containing the model weights.
        model_name (str): Name of the model to load.

    Returns:
        LlavaNextForConditionalGeneration: Configured model with LoRA applied.
    """
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = LlavaNextForConditionalGeneration.from_pretrained(
        pretrained_model_name_or_path=f"{weights_dir}/{model_name}",
        torch_dtype=torch.float16,
        quantization_config=bnb_config,
        device_map="auto",
    )

    # Apply LoRA
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=find_all_linear_names(model),
        init_lora_weights="gaussian",
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)

    return model


class LLaVAFineTuner(pl.LightningModule):
    """
    PyTorch Lightning module for fine-tuning LLaVA models.

    This module handles the training and validation process for LLaVA models,
    including LoRA fine-tuning and mixed-precision training.

    Attributes:
        model_name (str): Name of the pre-trained model to use.
        learning_rate (float): Learning rate for optimization.
        weight_decay (float): Weight decay for regularization.
        processor (LlavaNextProcessor): Processor for tokenizing and encoding inputs.
        model (LlavaNextForConditionalGeneration): The LLaVA model.
    """

    def __init__(self, hparams):
        """
        Initialize the LLaVAFineTuner.

        Args:
            hparams: Hyperparameters for the model and training process.
        """
        super().__init__()
        self.save_hyperparameters(hparams)

        self.model_name = self.hparams.model_name
        self.weights_dir = self.hparams.weights_dir
        self.learning_rate = self.hparams.learning_rate
        self.weight_decay = self.hparams.weight_decay
        self.processor_dir = self.hparams.processor_dir

        self.processor = load_processor(self.processor_dir)
        self.model = load_model(self.weights_dir, self.model_name)

    def _step(self, batch: Dict, prefix: str) -> torch.Tensor:
        """
        Perform a single training or validation step.

        Args:
            batch (Dict): The input batch.
            prefix (str): Prefix for logging ('train' or 'val').

        Returns:
            torch.Tensor: The loss for this step.
        """
        outputs = self.model(**batch)
        loss = outputs.loss
        current_lr = self.lr_schedulers().get_last_lr()[0]

        self.log(
            f"{prefix}_loss",
            loss,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            logger=True,
        )
        self.log(
            f"{prefix}_loss_epoch",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        self.log(
            "lr",
            current_lr,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

        return loss

    def training_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        """Perform a single training step."""
        return self._step(batch, prefix="train")

    def validation_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        """Perform a single validation step."""
        return self._step(batch, prefix="val")

    def configure_optimizers(self):
        """
        Configure optimizers and learning rate schedulers.

        Returns:
            Tuple[List, List]: A tuple containing a list of optimizers and a list of scheduler configurations.
        """
        optimizer = AdamW(
            self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )

        total_steps = self.trainer.estimated_stepping_batches
        warmup_steps = int(0.1 * total_steps)

        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def train_dataloader(self) -> DataLoader:
        """Create and return the training DataLoader."""
        train_dataset = XRayReportDataset(
            self.hparams.images_dir, self.hparams.annotation_file, split="train"
        )
        return DataLoader(
            train_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            collate_fn=lambda b: XRayReportDataset.collate_fn(b, self.processor),
            num_workers=self.hparams.num_workers,
            persistent_workers=True,
        )

    def val_dataloader(self) -> DataLoader:
        """Create and return the validation DataLoader."""
        val_dataset = XRayReportDataset(
            self.hparams.images_dir, self.hparams.annotation_file, split="val"
        )
        return DataLoader(
            val_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            collate_fn=lambda b: XRayReportDataset.collate_fn(b, self.processor),
            num_workers=self.hparams.num_workers,
            persistent_workers=True,
        )


def main(args: ArgumentParser):
    """
    Main function to run the LLaVA fine-tuning process.

    Args:
        args (ArgumentParser): Parsed command-line arguments.
    """
    # Set random seed and environment variables
    pl.seed_everything(args.seed)
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    torch.cuda.empty_cache()
    torch.set_float32_matmul_precision("medium")

    model = LLaVAFineTuner(args)

    # Setup callbacks and logger
    callbacks = [
        ModelCheckpoint(
            dirpath=args.checkpoint_dir,
            filename="llava-{epoch:02d}-{val_loss_epoch:.2f}",
            save_top_k=2,
            every_n_epochs=1,
            monitor="val_loss_epoch",
            mode="min",
            verbose=True,
        ),
        ModelCheckpoint(
            dirpath=args.checkpoint_dir,
            filename="llava-{epoch:02d}-{val_loss_epoch:.2f}",
            save_top_k=1,
            every_n_epochs=1,
            save_last=True,
            verbose=True,
        ),
        LearningRateMonitor(logging_interval="step"),
    ]
    logger = WandbLogger(
        project="llava-finetune",
        save_dir=args.log_dir,
        entity="adibvafa",
        resume="allow",
    )

    # Initialize the PyTorch Lightning Trainer
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="gpu",
        devices=args.num_gpus,
        strategy="deepspeed",
        precision="bf16-mixed",
        callbacks=callbacks,
        logger=logger,
        deterministic=False,
        enable_checkpointing=True,
        enable_progress_bar=True,
        enable_model_summary=True,
        log_every_n_steps=5,
        accumulate_grad_batches=args.gradient_accumulation_steps,
        gradient_clip_val=1.0,
    )

    # Start the training process
    trainer.fit(model)


if __name__ == "__main__":
    parser = ArgumentParser()

    # Add command-line arguments
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--log_dir", type=str, default="logs")
    parser.add_argument("--images_dir", type=str, required=True)
    parser.add_argument("--annotation_file", type=str, required=True)
    parser.add_argument("--weights_dir", type=str, default=".cache")
    parser.add_argument("--processor_dir", type=str, default=".cache")
    parser.add_argument("--model_name", type=str, default="llava-v1.6-mistral-7b-hf")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_epochs", type=int, default=5)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=250)

    args = parser.parse_args()

    main(args)
