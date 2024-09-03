# Finetuning LlaVA-Next on IU X-Ray Dataset
The objectives of this repository are described in `XRay-ReportGeneration.pdf`. 

![image](https://github.com/user-attachments/assets/58dc03ff-1f63-4a8a-9866-d534e1ff130c)
<br><br>

## Environments and Requirements
- OS: Linux, Ubuntu, 18.04.6 LTS
- GPU: NVIDIA A100-80GB
- CUDA: 12
- Python: 3.11

To install requirements:

```setup
pip install -r requirements.txt
```
<br>

## Dataset

- Dataset is available here https://paperswithcode.com/dataset/iu-x-ray
- The annotations are provided in the repo under `results`.
<br><br>


## Task 1

This task uses OpenAI API with structured outputs to break down a radiology report to findings of lung, heart, mediastinal, bone, and others.
The code used is available at `task1.ipynb`. The resulting break downs for validation data are available at `results/annotations.json`.
<br><br>


## Task 2

This task uses the `finetune.py` script to fientune the Llava-Next model on the IU X-Ray dataset.
Then, `inference.py` is used to generated radiology reports on validation and test datasets, along with evaluating model performance using GREEN metric.
<br><br>


### Finetuning Llava-Next

The `finetune.py` script uses QLoRA and PyTorch Lightning to finetune the llava-v1.6-mistral-7b-hf model.
Only the linear layers of the model are finetuned on the 2069 training examples provided in train dataset.

```python
python finetune.py \
    --images_dir images_dir \
    --annotation_file annotation_file \
    --weights_dir weights_dir \
    --processor_dir "llava-hf/llava-v1.6-mistral-7b-hf" \
    --checkpoint_dir checkpoint_dir \
    --log_dir log_dir \
    --batch_size 1 \
    --max_epochs 4 \
    --learning_rate 5e-5 \
    --num_gpus 1 \
    --num_workers 4 \
    --gradient_accumulation_steps 8
```

### Inference and Evaluation

The `inference.py` script runs batch inference on validation and test datasets, saving generated reports in the `results/annotations.json` file.
It evaluates generated reports using the GREEN score.

```python
python inference.py \
    --checkpoint_path checkpoint_path \
    --cache_dir cache_dir \
    --weights_dir weights_dir \
    --processor_dir "llava-hf/llava-v1.6-mistral-7b-hf" \
    --images_dir images_dir \
    --annotation_file annotation_file \
    --output_dir output_dir \
    --batch_size 12 \
    --max_new_tokens 200
```
<br>

## Trained Models
Trained model will be provided upon approval!
<br><br>

## Results

Model performance is as follows:

| Data Split | Lung | Heart | Mediastinal | Bone |
|------------|------|-------|-------------|------|
| Test | 0.70    | 0.89     | 0.66           | 0.32    |
| Valid    |  0.69   | 0.72     | 0.50           | 0.31    |
<br>

## Acknowledgement
We thank the WangLab for this interesting project.
