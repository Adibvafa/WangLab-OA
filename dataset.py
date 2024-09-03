import os
import json
from typing import List, Dict

from PIL import Image

from torch.utils.data import Dataset
from transformers import LlavaNextProcessor


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

    def __init__(
        self,
        data_dir: str,
        annotation_file: str,
        split: str = "train",
        with_assistant: bool = True,
        max_images: int = 2,
    ) -> None:
        """
        Initialize the XRayReportDataset.

        Args:
            data_dir (str): Directory containing the X-ray images.
            annotation_file (str): Path to the JSON file containing annotations.
            split (str, optional): Dataset split ('train' or 'val'). Defaults to 'train'.
            with_assistant (bool, optional): Whether to include the assistant response. Defaults to True.
            max_images (int, optional): Maximum number of images to include in the conversation. Defaults to 2.
        """
        self.data_dir = data_dir
        self.split = split
        self.with_assistant = with_assistant
        self.max_images = max_images

        with open(annotation_file, "r") as f:
            self.annotations = json.load(f)[split]

        self.regions = ["lung", "heart", "mediastinal", "bone"]

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
        patient_id = annotation["id"]
        report = self.prepare_report(annotation["report"])

        image_folder = os.path.join(self.data_dir, patient_id)
        image_files = sorted(
            [f for f in os.listdir(image_folder) if f.endswith(".png")]
        )
        images = [
            Image.open(os.path.join(image_folder, img_file)).convert("RGB")
            for img_file in image_files
        ]
        images = images[: self.max_images]  # Limit to first two images

        conversation = self.prepare_conversation(report, len(images))

        return {
            "patient_id": patient_id,
            "images": images,
            "report": report,
            "conversation": conversation,
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
        user = {
            "role": "user",
            "content": [{"type": "image"}] * num_images
            + [
                {
                    "type": "text",
                    "text": "Generate an X-ray report for Lung, Heart, Mediastinal, and Bone.",
                }
            ],
        }

        assistant = {
            "role": "assistant",
            "content": [{"type": "text", "text": report}],
        }

        return [user, assistant] if self.with_assistant else [user]

    def prepare_report(self, report: Dict[str, str]) -> str:
        """
        Prepare a formatted report string from the annotation dictionary.

        Args:
            report (Dict[str, str]): A dictionary containing report sections.

        Returns:
            str: A formatted report string.
        """
        report_text = ""
        for region in self.regions:
            report_text += f"\n{region.capitalize()}: {report.get(region, 'NA')}"
        return report_text

    @staticmethod
    def collate_fn(
        batch: List[Dict],
        processor: LlavaNextProcessor,
        add_generation_prompt: bool = False,
        max_length: int = 250,
    ) -> Dict:
        """
        Collate function for DataLoader.

        Args:
            batch (List[Dict]): A list of dictionaries, each representing a batch item.
            processor (LlavaNextProcessor): The processor for tokenizing and encoding inputs.

        Returns:
            Dict: A dictionary of tensors ready for model input.
        """
        images = [img for item in batch for img in item["images"]]
        prompts = [
            processor.apply_chat_template(
                item["conversation"], add_generation_prompt=add_generation_prompt
            )
            for item in batch
        ]

        inputs = processor(
            text=prompts,
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )

        labels = inputs["input_ids"].clone()
        labels[labels == processor.tokenizer.pad_token_id] = -100
        inputs["labels"] = labels

        return inputs
