#!/usr/bin/env bash
set -euo pipefail

: "${STUDENT_MODEL:?Set STUDENT_MODEL to a local path or Hugging Face model id.}"
: "${TRAIN_DATA:?Set TRAIN_DATA to a JSON/JSONL training file or dataset path.}"

OUTPUT_DIR="${OUTPUT_DIR:-./outputs/sft}"
NUM_GPUS="${NUM_GPUS:-1}"

torchrun --nproc_per_node="${NUM_GPUS}" \
  llm_hybrid_kd/train_scripts/train_sft.py \
  --model_path "${STUDENT_MODEL}" \
  --data_path "${TRAIN_DATA}" \
  --output_dir "${OUTPUT_DIR}" \
  --learning_rate "${LEARNING_RATE:-1e-5}" \
  --per_device_train_batch_size "${BATCH_SIZE:-4}" \
  --gradient_accumulation_steps "${GRAD_ACC:-8}" \
  --num_train_epochs "${EPOCHS:-2}" \
  --max_length "${MAX_LENGTH:-2048}" \
  --use_lora \
  --use_deepspeed
