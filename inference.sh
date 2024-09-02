#!/bin/bash
#SBATCH --job-name=llava_inference
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
CHECKPOINT_DIR="${SCRATCH}/WangLab/checkpoints"
OUTPUT_DIR="${PROJECT_DIR}/inference"
CACHE_DIR="${SCRATCH}/.cache"

# Activate virtual environment
source "${ROOT}/light/bin/activate"

# Change to project directory
cd "${PROJECT_DIR}"

# Set environment variables for better debugging and performance
export CUBLAS_WORKSPACE_CONFIG=:4096:2
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1
export PYTHONUNbBUFFERED=1

# Run the Python script with unbuffered output
stdbuf -oL -eL srun python3 inference.py \
    --checkpoint_path "${CHECKPOINT_DIR}/llava-epoch=02-val_loss_epoch=0.43.ckpt/pytorch_model.bin" \
    --cache_dir "${CACHE_DIR}" \
    --weights_dir "${WEIGHTS_DIR}" \
    --processor_dir "llava-hf/llava-v1.6-mistral-7b-hf" \
    --images_dir "${DATA_DIR}/images" \
    --annotation_file "${DATA_DIR}/annotation_quiz_all_with_val.json" \
    --output_dir "${OUTPUT_DIR}" \
    --batch_size 12 \
    --max_new_tokens 200