#!/bin/bash
#SBATCH --job-name=llava_finetune
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --partition=a100
#SBATCH --qos=a100_amritk
#SBATCH --cpus-per-task=8
#SBATCH --mem=48GB
#SBATCH --time=12:00:00
#SBATCH --no-requeue

# Define base directories
ROOT="/h/afallah"
BASE_DIR="${ROOT}/WangLab"

# Project-specific directories
PROJECT_DIR="${BASE_DIR}/OA"
DATA_DIR="${SCRATCH}/WangLab/OA/data"
WEIGHTS_DIR="/model-weights"
CHECKPOINT_DIR="${SCRATCH}/WangLab/new_checkpoints"
LOG_DIR="${PROJECT_DIR}/logs"

# Activate virtual environment
source "${ROOT}/light/bin/activate"

# Change to project directory
cd "${PROJECT_DIR}"

# Set environment variables for better debugging and performance
export CUBLAS_WORKSPACE_CONFIG=:4096:2
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1
export PYTHONUNBUFFERED=1

# Run the Python script with unbuffered output
stdbuf -oL -eL srun python3 finetune.py \
    --images_dir "${DATA_DIR}/images" \
    --annotation_file "${DATA_DIR}/annotation_quiz_all_with_val.json" \
    --weights_dir "${WEIGHTS_DIR}" \
    --processor_dir "llava-hf/llava-v1.6-mistral-7b-hf" \
    --checkpoint_dir "${CHECKPOINT_DIR}" \
    --log_dir "${LOG_DIR}" \
    --batch_size 1 \
    --max_epochs 4 \
    --learning_rate 5e-5 \
    --num_gpus 1 \
    --num_workers 4 \
    --gradient_accumulation_steps 8
