import re
import os
import sys
import json
import argparse
from tqdm import tqdm
from typing import List, Dict, Tuple

import pandas as pd
from green_score.green import compute

import torch
from torch.utils.data import DataLoader
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration

from dataset import XRayReportDataset
from finetune import load_processor, load_model


def load_model_and_processor(args: argparse.Namespace) -> Tuple[LlavaNextForConditionalGeneration, LlavaNextProcessor]:
    """
    Load the LLaVA model and processor with the specified checkpoint.

    Args:
        args: Command-line arguments containing model configuration.

    Returns:
        Tuple of loaded model and processor.
    """
    # Load model and processor
    processor = load_processor(args.processor_dir, padding_side="left")
    model = load_model(args.weights_dir, args.model_name)
    
    # Load the fine-tuned weights
    state_dict = torch.load(args.checkpoint_path, map_location="cpu")
    state_dict_fixed = {key[6:]: value for key, value in state_dict.items()}
    model.load_state_dict(state_dict_fixed)
    
    model.eval()
    return model, processor


def generate_reports(model: LlavaNextForConditionalGeneration, 
                     processor: LlavaNextProcessor, 
                     dataloader: DataLoader, 
                     device: torch.device,
                     max_new_tokens: int = 250) -> List[Dict[str, str]]:
    """
    Generate reports for the given dataset and parse them into sections.

    Args:
        model: The LLaVA model.
        processor: The LLaVA processor.
        dataloader: DataLoader for the dataset.
        device: Device to run the model on.
        max_new_tokens: Maximum number of new tokens to generate.

    Returns:
        List containing generated reports parsed into sections.
    """
    generated_reports = []
    model.to(device)
    
    for batch in tqdm(dataloader, desc="Generating Reports", unit="batch", total=len(dataloader)):

        with torch.no_grad():
            output = model.generate(
                input_ids=batch['input_ids'].to(device),
                attention_mask=batch['attention_mask'].to(device),
                pixel_values=batch['pixel_values'].to(device),
                image_sizes=batch['image_sizes'].to(device),
                max_new_tokens=max_new_tokens
            )
        
        for i in range(len(output)):
            decoded_output = processor.decode(output[i], skip_special_tokens=True)
            report_start = decoded_output.find("[/INST]") + len("[/INST]")
            report_end = decoded_output.find("<\s>")
            generated_report = decoded_output[report_start:report_end].strip()
            parsed_report = parse_report_sections(generated_report)
            generated_reports.append(parsed_report)
        
        # break #DEBUG
        
    return generated_reports


def parse_report_sections(report: str) -> Dict[str, str]:
    """
    Parse the report into sections.
    
    Args:
        report: The full report string.
    
    Returns:
        A dictionary with keys 'lung', 'heart', 'mediastinal', 'bone', and 'full'.
    """
    sections = {
        'lung': '',
        'heart': '',
        'mediastinal': '',
        'bone': '',
        'full': report.strip()
    }
    
    patterns = {
        'lung': r'Lung:(.+?)(?=Heart:|Mediastinal:|Bone:|$)',
        'heart': r'Heart:(.+?)(?=Lung:|Mediastinal:|Bone:|$)',
        'mediastinal': r'Mediastinal:(.+?)(?=Lung:|Heart:|Bone:|$)',
        'bone': r'Bone:(.+?)(?=Lung:|Heart:|Mediastinal:|$)'
    }
    
    for section, pattern in patterns.items():
        match = re.search(pattern, report, re.DOTALL | re.IGNORECASE)
        if match:
            sections[section] = match.group(1).strip()
    
    return sections


def update_annotation_file(annotation_file: str, split: str, patient_ids: List[str], generated_reports: List[str]):
    """
    Update the annotation file with generated reports.

    Args:
        annotation_file: Path to the annotation file.
        split: Dataset split ('val' or 'test').
        patient_ids: List of patient IDs.
        generated_reports: List of generated reports.
    """
    with open(annotation_file, 'r') as f:
        annotations = json.load(f)
    
    for patient_id, report in zip(patient_ids, generated_reports):
        for item in annotations[split]:
            if item['id'] == patient_id:
                item['generated_report'] = report
    
    with open(annotation_file, 'w') as f:
        json.dump(annotations, f, indent=2)


def run_green_evaluation(refs: List[Dict[str, str]], hyps: List[Dict[str, str]], 
                         output_dir: str, cache_dir: str, batch_size: int = 4) -> pd.DataFrame:
    """
    Run GREEN evaluation on the generated reports for each section and overall.

    Args:
        refs: List of dictionaries containing reference (ground truth) reports sections.
        hyps: List of dictionaries containing generated (hypothesis) reports sections.
        output_dir: Directory to save output files.
        cache_dir: Cache directory for GREEN model.
        batch_size: Batch size for evaluation.

    Returns:
        DataFrame containing evaluation results for each section and overall.
    """
    model_name = "StanfordAIMI/GREEN-radllama2-7b"
    sections = ['full', 'lung', 'heart', 'mediastinal', 'bone']
    results = []

    for section in sections:
        section_refs = [ref[section] for ref in refs]
        section_hyps = [hyp[section] for hyp in hyps]
        
        section_output_dir = os.path.join(output_dir, section)
        os.makedirs(section_output_dir, exist_ok=True)

        try:
            compute(model_name, section_refs, section_hyps, output_dir=section_output_dir, 
                    cache_dir=cache_dir, batch_size=batch_size)
            section_results = pd.read_csv(f"{section_output_dir}/results_.csv")
            section_results = section_results.add_prefix(f"{section}_")
            results.append(section_results)

        except Exception as e:
            print(f"Error in GREEN evaluation for {section} section: {str(e)}", file=sys.stderr)
    
    # Combine results
    combined_results = pd.concat(results, axis=1)
    return combined_results


def main(args: argparse.Namespace):
    """
    Main function to run the X-ray report generation and evaluation process.

    Args:
        args: Parsed command-line arguments.
    """
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    torch.cuda.empty_cache()
    torch.set_float32_matmul_precision("medium")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model and processor
    model, processor = load_model_and_processor(args)

    for split in ['test', 'val']:
        # Load the dataset and dataloader
        print(f"\nRunning evaluation for {split} set...\n")
        dataset = XRayReportDataset(args.images_dir, args.annotation_file, split=split, with_assistant=False)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                                collate_fn=lambda b: XRayReportDataset.collate_fn(b, processor, add_generation_prompt=True))
        
        # Generate reports using model checkpoint
        generated_reports = generate_reports(
            model=model,
            processor=processor,
            dataloader=dataloader,
            device=device,
            max_new_tokens=args.max_new_tokens
        )
        patient_ids = [annotation['id'] for annotation in dataset.annotations]
        print(f"\nGenerated {len(generated_reports)} reports for {split} set.\n")

        # Parse ground truth reports
        actual_reports = [parse_report_sections(sample['report'].strip()) for sample in dataset]

        # Update annotation file with generated reports (full reports only)
        update_annotation_file(args.annotation_file, split, patient_ids, [report['full'] for report in generated_reports])
        print(f"\nUpdated annotation file with generated reports for {split} set.\n")

        # Free GPU memory
        model = model.to("cpu")

        # Run GREEN evaluation and save results
        output_dir = f"{args.output_dir}/{split}"
        os.makedirs(output_dir, exist_ok=True)
        #DEBUG
        results_df = run_green_evaluation(
            refs=actual_reports, #[:len(generated_reports)], 
            hyps=generated_reports,
            output_dir=output_dir,
            cache_dir=args.cache_dir,
            batch_size=args.batch_size
        )

        #DEBUG
        results_df['patient_ids'] = patient_ids#[:len(generated_reports)]
        results_df.to_csv(f"{output_dir}/results_detailed.csv", index=False)

        print(f"\nCompleted evaluation for {split} set. Detailed results saved in {output_dir}/results_detailed.csv\n")

        # For backwards compatibility, save overall results in the original format
        overall_results = results_df[['patient_ids'] + [col for col in results_df.columns if col.startswith('full_')]]
        overall_results.columns = [col.replace('full_', '') for col in overall_results.columns]
        overall_results.to_csv(f"{output_dir}/results_.csv", index=False)

        print(f"Overall results saved in {output_dir}/results_.csv\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="X-Ray Report Generation and Evaluation")

    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to the model checkpoint")
    parser.add_argument("--cache_dir", type=str, default=".cache", help="Cache directory for GREEN model")
    parser.add_argument('--weights_dir', type=str, default='.cache')
    parser.add_argument('--processor_dir', type=str, default='.cache')
    parser.add_argument('--model_name', type=str, default='llava-v1.6-mistral-7b-hf') 
    parser.add_argument("--images_dir", type=str, required=True, help="Directory containing X-ray images")
    parser.add_argument("--annotation_file", type=str, required=True, help="Path to the annotation file")
    parser.add_argument("--output_dir", type=str, default="inference", help="Directory to save output files")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for inference")
    parser.add_argument("--max_new_tokens", type=int, default=250, help="Maximum number of new tokens to generate")
    
    args = parser.parse_args()
    main(args)