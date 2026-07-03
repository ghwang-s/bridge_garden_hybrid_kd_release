import os
import argparse
from pathlib import Path
import importlib.util

os.environ['GLOO_SOCKET_IFNAME'] = 'eth0'
os.environ['NCCL_SOCKET_IFNAME'] = 'eth0'
os.environ['NCCL_IB_DISABLE'] = '1'
os.environ['NCCL_SOCKET_FAMILY'] = 'AF_INET'

os.environ['NCCL_NET'] = 'Socket'
os.environ['NCCL_P2P_LEVEL'] = 'NVL'

from trl import HybridGKDTrainer
from trl.trainer.gkd_config import GKDConfig
from trl.trainer.utils import DataCollatorForGKDPromptCompletion
from datasets import load_dataset
from datetime import timedelta
import json
import math
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig

# DeepSpeed ZeRO-2 configuration
DEEPSPEED_CONFIG = {
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto",
    "gradient_accumulation_steps": "auto",
    "gradient_clipping": 1.0,
    "steps_per_print": 200,
    "bf16": {
        "enabled": "auto"
    },
    "fp16": {
        "enabled": "auto"
    },
    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {
            "device": "none",
            "pin_memory": True
        },
        "offload_param": {
            "device": "none"
        },
        "allgather_partitions": True,
        "allgather_bucket_size": 2e8,
        "overlap_comm": True,
        "reduce_scatter": True,
        "reduce_bucket_size": 2e8,
        "contiguous_gradients": True
    },
    "wall_clock_breakdown": False
}

DEEPSPEED_CONFIG_MEMORY_EFFICIENT = {
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto",
    "gradient_accumulation_steps": "auto",
    "gradient_clipping": 1.0,
    "bf16": {
        "enabled": "auto"
    },
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": True
        },
        "offload_param": {
            "device": "cpu",
            "pin_memory": True
        },
        "overlap_comm": True,
        "contiguous_gradients": True,
        "sub_group_size": 1e9,
        "reduce_bucket_size": "auto",
        "stage3_prefetch_bucket_size": "auto",
        "stage3_param_persistence_threshold": "auto",
        "stage3_max_live_parameters": 1e9,
        "stage3_max_reuse_distance": 1e9,
        "stage3_gather_16bit_weights_on_model_save": True,
    },
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": "auto",
            "betas": "auto",
            "eps": "auto",
            "weight_decay": "auto"
        }
    }
}

def parse_args():
    parser = argparse.ArgumentParser(description='Hybrid GKD Training with DeepSpeed')
    

    parser.add_argument('--output_dir', type=str, default='./outputs/hybrid_gkd')
    parser.add_argument('--model_path', type=str, required=True, help='Student model path or Hugging Face model id')
    parser.add_argument('--teacher_model_path', type=str, required=True, help='Teacher model path (required for GKD)')
    parser.add_argument('--data_path', type=str, required=True, help='Training dataset path or dataset id')
    

    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument('--per_device_train_batch_size', type=int, default=4)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=32)
    parser.add_argument('--num_train_epochs', type=int, default=2)
    parser.add_argument('--max_length', type=int, default=2048)
    parser.add_argument('--seed', type=int, default=42)
    

    parser.add_argument('--use_lora', action='store_true', help='Enable LoRA')
    parser.add_argument('--lora_r', type=int, default=16)
    parser.add_argument('--lora_alpha', type=int, default=32)
    

    parser.add_argument('--lmbda', type=float, default=1.0, help='Weight for on-policy student generations')
    parser.add_argument('--beta', type=float, default=0.5, help='Interpolation coefficient for JSD (0-1)')
    parser.add_argument('--temperature', type=float, default=1.0, help='Temperature for generation and distillation')
    parser.add_argument('--max_new_tokens', type=int, default=512, help='Maximum new tokens to generate')
    parser.add_argument('--lambda_fkl', type=float, default=1.0, help='Weight for the soft KD loss term')
    parser.add_argument('--lambda_sft', type=float, default=1.0, help='Weight for the teacher-argmax hard loss term')
    

    parser.add_argument('--use_deepspeed', action='store_true', help='Enable DeepSpeed')
    parser.add_argument('--memory_efficient', action='store_true', help='Use memory-efficient DeepSpeed config (ZeRO-3 + CPU offload)')
    
    return parser.parse_args()

def save_deepspeed_config(config_dict, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    config_path = os.path.join(output_dir, "ds_config.json")
    with open(config_path, 'w') as f:
        json.dump(config_dict, f, indent=2)
    print(f"DeepSpeed config saved to: {config_path}")
    return config_path

def init_distributed_training():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    
    import torch.distributed as dist
    

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")
    
    if not dist.is_initialized():
        dist.init_process_group(
            backend='nccl',
            timeout=timedelta(seconds=7200)
        )
        dist.barrier()
        if rank == 0:
            print(f" Process group initialized with 7200s timeout")
    
    print(f" Distributed Training Environment:")
    print(f"   - RANK: {rank}, LOCAL_RANK: {local_rank}, WORLD_SIZE: {world_size}")
    print(f"   - Device: {device}")
    
    return local_rank, world_size, rank, device

def main():
    args = parse_args()

    if args.use_deepspeed and importlib.util.find_spec("deepspeed") is None:
        raise ImportError(
            "`--use_deepspeed` was passed, but deepspeed is not installed in the current environment."
        )
    
    local_rank, world_size, rank, device = init_distributed_training()
    is_main_process = local_rank == 0

    output_dir = (
        f"{args.output_dir}/learning_rate_{args.learning_rate}_lmbda_{args.lmbda}_beta_{args.beta}"
        f"_lambda_fkl_{args.lambda_fkl}_lambda_sft_{args.lambda_sft}"
    )
    logging_dir = f"{output_dir}/logs"

    if is_main_process:
        ds_config_path = os.path.join(output_dir, "ds_config.json")
        if os.path.exists(ds_config_path):
            try:
                os.remove(ds_config_path)
            except OSError as e:
                print(f"Could not remove existing DeepSpeed config: {e}")

    deepspeed_config = None
    if args.use_deepspeed:
        if args.memory_efficient:
            deepspeed_config = DEEPSPEED_CONFIG_MEMORY_EFFICIENT
            config_name = "ZeRO-3 + CPU Offload"
        else:
            deepspeed_config = DEEPSPEED_CONFIG
            config_name = f"ZeRO-{deepspeed_config['zero_optimization']['stage']}"
        
        deepspeed_config["gradient_accumulation_steps"] = args.gradient_accumulation_steps
        deepspeed_config["train_batch_size"] = "auto"
        deepspeed_config["train_micro_batch_size_per_gpu"] = args.per_device_train_batch_size
        
        if is_main_process:
            print(f"Using DeepSpeed config: {config_name}")
    
    if is_main_process:
        print("\n" + "="*60)
        print("Hybrid GKD Training Configuration:")
        print("="*60)
        print(f"Student Model: {args.model_path}")
        print(f"Teacher Model: {args.teacher_model_path}")
        print(f"Data Path: {args.data_path}")
        print(f"Output Dir: {output_dir}")
        print(f"Learning Rate: {args.learning_rate}")
        print(f"Batch Size per Device: {args.per_device_train_batch_size}")
        print(f"Gradient Accumulation: {args.gradient_accumulation_steps}")
        print(f"World Size: {world_size}")
        print(f"DeepSpeed: {'Enabled' if args.use_deepspeed else 'Disabled'}")
        print(f"Epochs: {args.num_train_epochs}")
        print(f"Max Length: {args.max_length}")
        print(f"LoRA: {'Enabled' if args.use_lora else 'Disabled'} (r: {args.lora_r}, alpha: {args.lora_alpha})")
        print(f"GKD Lambda: {args.lmbda}, Beta: {args.beta}, Temperature: {args.temperature}")
        print(f"Hybrid Loss Weights: lambda_fkl={args.lambda_fkl}, lambda_sft={args.lambda_sft}")
        
        if args.use_deepspeed:
            print(f"DeepSpeed Config: {config_name}")
        print("="*60 + "\n")


    if is_main_process:
        print(" Loading and processing dataset...")
    
    raw_dataset = load_dataset("json", data_files={"train": args.data_path}, split="train", streaming=False)
    
    def convert_to_prompt_completion_format(examples):
        prompts = []
        completions = []
        
        for i in range(len(examples['prompt'])):

            prompts.append(examples['prompt'][i])
            completions.append(examples['completion'][i])
        
        return {
            "prompt": prompts,
            "completion": completions
        }
    

    with GKDConfig(output_dir=output_dir).main_process_first(desc="dataset map pre-processing"):
        dataset = raw_dataset.map(
            convert_to_prompt_completion_format,
            batched=True,
            remove_columns=raw_dataset.column_names,
            load_from_cache_file=True,
            desc="Converting to prompt-completion format for GKD"
        )

    if is_main_process:
        print(f" Dataset processing complete. Total samples: {len(dataset)}")
    

    total_samples = len(dataset)
    if is_main_process:
        print(f" Total dataset size: {total_samples} samples")
        print(f" Each of {world_size} processes will handle ~{total_samples // world_size} samples")
    

    EFFECTIVE_BATCH_SIZE = args.per_device_train_batch_size * args.gradient_accumulation_steps * world_size
    MAX_STEPS_CALCULATED = max(1, math.ceil((total_samples * args.num_train_epochs) / EFFECTIVE_BATCH_SIZE))
    
    if is_main_process:
        print(f" Effective batch size: {EFFECTIVE_BATCH_SIZE} (per_device={args.per_device_train_batch_size} × grad_accum={args.gradient_accumulation_steps} × world_size={world_size})")
        print(f" Calculated max steps: {MAX_STEPS_CALCULATED}")
    

    peft_config = None
    if args.use_lora:
        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )
    

    if is_main_process:
        print(f"Loading student model from {args.model_path}...")
    
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        use_fast=True,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        if is_main_process:
            print(f" Pad token set to eos_token: {tokenizer.pad_token}")
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype='auto',
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    
    if is_main_process:
        print("\n" + "="*60)
        print("Preparing Teacher Model")
        print("="*60)
        print(f"Teacher model path: {args.teacher_model_path}")
        print("Teacher model will be instantiated inside HybridGKDTrainer.\n")
    

    data_collator = DataCollatorForGKDPromptCompletion(
        tokenizer=tokenizer,
        max_length=args.max_length,
        completion_only_loss=True,
    )
    
    if is_main_process:
        print("Creating HybridGKDTrainer...")
    

    training_args = GKDConfig(
        output_dir=output_dir,
        logging_dir=logging_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        logging_strategy="steps",
        logging_steps=2,
        save_strategy="epoch",
        save_total_limit=5,
        dataloader_drop_last=False,
        bf16=True,
        dataloader_num_workers=0,
        num_train_epochs=args.num_train_epochs,
        lr_scheduler_type="cosine",
        learning_rate=args.learning_rate,
        warmup_ratio=0.03,
        seed=args.seed,
        optim="adamw_torch",
        ddp_timeout=10800,
        
        report_to='tensorboard' if is_main_process else None,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        ddp_find_unused_parameters=False,
        max_length=args.max_length,
        

        lmbda=args.lmbda,
        beta=args.beta,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        lambda_fkl=args.lambda_fkl,
        lambda_sft=args.lambda_sft,
        teacher_model_name_or_path=args.teacher_model_path,
        teacher_model_init_kwargs={
            "dtype": args.teacher_dtype,
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
        },
        

        deepspeed=deepspeed_config if args.use_deepspeed else None,
    )

    if is_main_process:
        print(f"\nModel output_dir: {training_args.output_dir}")
        print(f"Logging dir: {training_args.logging_dir}")
        if args.use_deepspeed:
            print(f"DeepSpeed config: {deepspeed_config}")
        print()
    

    trainer = HybridGKDTrainer(
        model=model,
        teacher_model=args.teacher_model_path,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    
    if is_main_process:
        print(f"\nStarting Hybrid GKD training...\n")
    
    trainer.train()
    

    if is_main_process:
        print(f"\nSaving final model to {training_args.output_dir}...")
    
    trainer.save_model(training_args.output_dir)
    
    if is_main_process:
        print(f"\nTraining completed!")
        print(f"Model saved to: {training_args.output_dir}")

if __name__ == "__main__":
    main()
