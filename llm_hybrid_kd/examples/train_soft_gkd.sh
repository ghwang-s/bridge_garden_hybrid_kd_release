#!/usr/bin/env bash
set -euo pipefail

: "${STUDENT_MODEL:?Set STUDENT_MODEL to a local path or Hugging Face model id.}"
: "${TEACHER_MODEL:?Set TEACHER_MODEL to a local path or Hugging Face model id.}"
: "${TRAIN_DATA:?Set TRAIN_DATA to a JSON/JSONL training file or dataset path.}"

OUTPUT_DIR="${OUTPUT_DIR:-./outputs/gkd}"
NUM_GPUS="${NUM_GPUS:-1}"

torchrun --nproc_per_node="${NUM_GPUS}" \
  llm_hybrid_kd/train_scripts/train_gkd.py \
  --model_path "${STUDENT_MODEL}" \
  --teacher_model_path "${TEACHER_MODEL}" \
  --data_path "${TRAIN_DATA}" \
  --output_dir "${OUTPUT_DIR}" \
  --learning_rate "${LEARNING_RATE:-5e-5}" \
  --per_device_train_batch_size "${BATCH_SIZE:-4}" \
  --gradient_accumulation_steps "${GRAD_ACC:-8}" \
  --num_train_epochs "${EPOCHS:-2}" \
  --max_length "${MAX_LENGTH:-2048}" \
  --lmbda "${LMBDA:-0.0}" \
  --beta "${BETA:-0.5}" \
  --temperature "${TEMPERATURE:-1.0}" \
  --max_new_tokens "${MAX_NEW_TOKENS:-512}" \
  --lambda_fkl "${LAMBDA_FKL:-1.0}" \
  --lambda_sft "${LAMBDA_SFT:-0.0}" \
  --use_lora \
  --use_deepspeed
