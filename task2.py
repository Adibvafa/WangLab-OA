import os
import json
from argparse import ArgumentParser
from typing import List, Dict, Optional

from PIL import Image

import torch
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from transformers import get_cosine_schedule_with_warmup

from peft import get_peft_model, LoraConfig, prepare_model_for_kbit_training
from transformers import BitsAndBytesConfig, LlavaNextProcessor, LlavaNextForConditionalGeneration

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DeepSpeedStrategy


class XRayReportDataset(Dataset):
    """
    A custom dataset for X-ray report generation.

    This dataset loads X-ray images and their corresponding reports from a specified directory
    and annotation file.

    Attributes:
        data_dir (str): Directory containing the X-ray images.
        split (str): Dataset split ('train' or 'val').
        annotations (List[Dict]): List of annotation dictionaries.
        regions (List[str]): List of anatomical regions for reporting.
    """

    def __init__(self, data_dir: str, annotation_file: str, split: str = 'train') -> None:
        """
        Initialize the XRayReportDataset.

        Args:
            data_dir (str): Directory containing the X-ray images.
            annotation_file (str): Path to the JSON file containing annotations.
            split (str, optional): Dataset split ('train' or 'val'). Defaults to 'train'.
        """
        self.data_dir = data_dir
        self.split = split

        with open(annotation_file, 'r') as f:
            self.annotations = json.load(f)[split]

        self.regions = ['lung', 'heart', 'mediastinal', 'bone']

    def __len__(self) -> int:
        """Return the number of items in the dataset."""
        return len(self.annotations)

    def __getitem__(self, idx: int) -> Dict:
        """
        Get an item from the dataset.

        Args:
            idx (int): Index of the item to retrieve.

        Returns:
            Dict: A dictionary containing images, conversation, and patient_id.
        """
        annotation = self.annotations[idx]
        patient_id = annotation['id']
        report = self.prepare_report(annotation['report'])

        image_folder = os.path.join(self.data_dir, patient_id)
        image_files = sorted([f for f in os.listdir(image_folder) if f.endswith('.png')])
        images = [Image.open(os.path.join(image_folder, img_file)).convert('RGB') for img_file in image_files]
        images = images[:2]  # Limit to first two images

        conversation = self.prepare_conversation(report, len(images))

        return {
            'images': images,
            'conversation': conversation,
            'patient_id': patient_id,
        }

    def prepare_conversation(self, report: str, num_images: int) -> List[Dict]:
        """
        Prepare a conversation-like structure for the model input.

        Args:
            report (str): The prepared report text.
            num_images (int): Number of images in the conversation.

        Returns:
            List[Dict]: A list of dictionaries representing the conversation.
        """
        conversation = [
            {
                "role": "user",
                "content": [{"type": "image"}] * num_images + [
                    {"type": "text", "text": "Generate an X-ray report for Lung, Heart, Mediastinal, and Bone."}
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": report}
                ],
            }
        ]
        return conversation
    
    def prepare_report(self, report: Dict[str, str]) -> str:
        """
        Prepare a formatted report string from the annotation dictionary.

        Args:
            report (Dict[str, str]): A dictionary containing report sections.

        Returns:
            str: A formatted report string.
        """
        report_text = ''
        for region in self.regions:
            report_text += f"{region.capitalize()}: {report.get(region, 'NA')}\n"
        return report_text.strip()

    @staticmethod
    def collate_fn(batch: List[Dict], processor: LlavaNextProcessor) -> Dict:
        """
        Collate function for DataLoader.

        Args:
            batch (List[Dict]): A list of dictionaries, each representing a batch item.
            processor (LlavaNextProcessor): The processor for tokenizing and encoding inputs.

        Returns:
            Dict: A dictionary of tensors ready for model input.
        """
        images = [img for item in batch for img in item['images']]
        prompts = [
            processor.apply_chat_template(item['conversation'], add_generation_prompt=True)
            for item in batch
        ]
        
        inputs = processor(
            text=prompts,
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.max_length
        )

        labels = inputs['input_ids'].clone()
        labels[labels == processor.tokenizer.pad_token_id] = -100
        inputs['labels'] = labels
        
        return inputs


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
        self.learning_rate = self.hparams.learning_rate
        self.weight_decay = self.hparams.weight_decay

        self.processor = LlavaNextProcessor.from_pretrained(
            self.model_name,
            cache_dir=self.hparams.cache_dir,
        )
        self.processor.tokenizer.padding_side = "right"
        self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token
        
        # Configure the model with LoRA and 4-bit quantization
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )
        self.model = LlavaNextForConditionalGeneration.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            cache_dir=self.hparams.cache_dir,
            quantization_config=bnb_config,
            device_map="auto",
        )
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            target_modules=self.find_all_linear_names(self.model),
            init_lora_weights="gaussian",
            bias="none",
            task_type="CAUSAL_LM",
        )
        self.model = prepare_model_for_kbit_training(self.model)
        self.model = get_peft_model(self.model, lora_config)
    
    def find_all_linear_names(self, model: torch.nn.Module) -> List[str]:
        """
        Find all linear layer names in the model for LoRA fine-tuning.

        Args:
            model (torch.nn.Module): The model to search.

        Returns:
            List[str]: A list of linear layer names.
        """
        lora_module_names = set()
        multimodal_keywords = ['multi_modal_projector', 'vision_model']

        for name, module in model.named_modules():
            if any(mm_keyword in name for mm_keyword in multimodal_keywords):
                continue
            if isinstance(module, torch.nn.Linear):
                names = name.split('.')
                lora_module_names.add(names[0] if len(names) == 1 else names[-1])
        
        if 'lm_head' in lora_module_names:
            lora_module_names.remove('lm_head')

        return list(lora_module_names)
    
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

        self.log(f"{prefix}_loss", loss, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        self.log(f"{prefix}_loss_epoch", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log("lr", current_lr, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        return loss

    def training_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        """Perform a single training step."""
        return self._step(batch, prefix='train')
    
    def validation_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        """Perform a single validation step."""
        return self._step(batch, prefix='val')

    def configure_optimizers(self):
        """
        Configure optimizers and learning rate schedulers.

        Returns:
            Tuple[List, List]: A tuple containing a list of optimizers and a list of scheduler configurations.
        """
        optimizer = AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
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
        train_dataset = XRayReportDataset(self.hparams.images_dir, self.hparams.annotation_file, split='train')
        return DataLoader(train_dataset, batch_size=self.hparams.batch_size, shuffle=True, 
                          collate_fn=lambda b: XRayReportDataset.collate_fn(b, self.processor), 
                          num_workers=self.hparams.num_workers, persistent_workers=True)

    def val_dataloader(self) -> DataLoader:
        """Create and return the validation DataLoader."""
        val_dataset = XRayReportDataset(self.hparams.images_dir, self.hparams.annotation_file, split='val')
        return DataLoader(val_dataset, batch_size=self.hparams.batch_size, shuffle=False, 
                          collate_fn=lambda b: XRayReportDataset.collate_fn(b, self.processor), 
                          num_workers=self.hparams.num_workers, persistent_workers=True)


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
            filename='llava-{epoch:02d}-{val_loss_epoch:.2f}',
            save_top_k=1,
            every_n_epochs=1,
            monitor='val_loss_epoch',
            mode='min',
            verbose=True
        ),
        LearningRateMonitor(logging_interval="step"),
    ]
    logger = WandbLogger(
        project='llava-finetune',
        save_dir=args.log_dir,
        entity='adibvafa',
        resume="allow",
    )

    # Initialize the PyTorch Lightning Trainer
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator='gpu',
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


if __name__ == '__main__':
    parser = ArgumentParser()

    # Add command-line arguments
    parser.add_argument('--seed', type=int, default=23)
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    parser.add_argument('--log_dir', type=str, default='logs')
    parser.add_argument('--images_dir', type=str, required=True)
    parser.add_argument('--annotation_file', type=str, required=True)
    parser.add_argument('--cache_dir', type=str, default='.cache')
    parser.add_argument('--model_name', type=str, default="llava-hf/llava-v1.6-mistral-7b-hf")    
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--max_epochs', type=int, default=5)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--num_gpus', type=int, default=1)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=8)
    parser.add_argument('--max_length', type=int, default=250)

    args = parser.parse_args()
    
    main(args)
