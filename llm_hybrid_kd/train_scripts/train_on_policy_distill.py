import os
import argparse
from pathlib import Path

os.environ['GLOO_SOCKET_IFNAME'] = 'eth0'
os.environ['NCCL_SOCKET_IFNAME'] = 'eth0'
os.environ['NCCL_IB_DISABLE'] = '1'
os.environ['NCCL_SOCKET_FAMILY'] = 'AF_INET'
os.environ['NCCL_NET'] = 'Socket'
os.environ['NCCL_P2P_LEVEL'] = 'NVL'

from trl import SFTConfig
from trl.trainer.on_policy_distill_config import OnPolicyDistillConfig
from trl.trainer.on_policy_distill_trainer import OnPolicyDistillTrainer
from trl.trainer.utils import SIMPLE_CHAT_TEMPLATE
from datasets import load_dataset
from datetime import timedelta
import json
import math
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig

try:
    import deepspeed
    DEEPSPEED_AVAILABLE = True
except ImportError:
    DEEPSPEED_AVAILABLE = False

DEEPSPEED_CONFIG = {
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto",
    "gradient_accumulation_steps": "auto",
    "gradient_clipping": 1.0,
    "steps_per_print": 200,
    "bf16": {"enabled": "auto"},
    "fp16": {"enabled": "auto"},
    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {"device": "none", "pin_memory": True},
        "offload_param": {"device": "none"},
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
    "bf16": {"enabled": "auto"},
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {"device": "none", "pin_memory": True},
        "offload_param": {"device": "none", "pin_memory": True},
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
        "params": {"lr": "auto", "betas": "auto", "eps": "auto", "weight_decay": "auto"}
    }
}


def parse_args():
    parser = argparse.ArgumentParser(description='On-Policy Distillation with vLLM')

    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--teacher_model_path', type=str, required=True)
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument('--per_device_train_batch_size', type=int, default=4)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=32)
    parser.add_argument('--num_train_epochs', type=int, default=2)
    parser.add_argument('--max_steps', type=int, default=None)
    parser.add_argument('--num_samples', type=int, default=None)
    parser.add_argument('--max_length', type=int, default=2048)
    parser.add_argument('--use_lora', action='store_true')
    parser.add_argument('--lora_r', type=int, default=16)
    parser.add_argument('--lora_alpha', type=int, default=32)
    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument('--alpha', type=float, default=0.05)
    parser.add_argument('--gamma', type=float, default=0.9999)
    parser.add_argument('--distill_alpha', type=float, default=1.0)
    parser.add_argument('--distill_beta', type=float, default=0.0)
    parser.add_argument('--distill_weight', type=float, default=0.5)
    parser.add_argument('--phi_type', type=str, default=None)
    parser.add_argument('--div', type=str, default="ab")

    parser.add_argument('--lmbda', type=float, default=1.0)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--max_new_tokens', type=int, default=512)
    parser.add_argument('--top_p', type=float, default=1.0)
    parser.add_argument('--top_k', type=int, default=-1)
    parser.add_argument('--repetition_penalty', type=float, default=1.0)

    parser.add_argument('--vllm_gpu_memory_utilization', type=float, default=0.5)
    parser.add_argument('--vllm_tensor_parallel_size', type=int, default=1)
    parser.add_argument('--vllm_dtype', type=str, default='auto')
    parser.add_argument('--vllm_max_model_len', type=int, default=None)
    parser.add_argument('--vllm_device', type=str, default=None)

    parser.add_argument('--data_format', type=str, default='auto',
                       choices=['auto', 'prompt_completion', 'messages_response', 'messages'])

    parser.add_argument('--use_deepspeed', action='store_true')
    parser.add_argument('--memory_efficient', action='store_true')

    return parser.parse_args()


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
        dist.init_process_group(backend='nccl', timeout=timedelta(seconds=7200))
        dist.barrier()

    return local_rank, world_size, rank, device


def main():
    args = parse_args()

    local_rank, world_size, rank, device = init_distributed_training()
    is_main_process = local_rank == 0

    output_dir = f"{args.output_dir}/learning_rate_{args.learning_rate}"
    logging_dir = f"{output_dir}/logs"

    deepspeed_config = None
    if args.use_deepspeed:
        if args.memory_efficient:
            deepspeed_config = DEEPSPEED_CONFIG_MEMORY_EFFICIENT
        else:
            deepspeed_config = DEEPSPEED_CONFIG
        deepspeed_config["gradient_accumulation_steps"] = args.gradient_accumulation_steps
        deepspeed_config["train_batch_size"] = "auto"
        deepspeed_config["train_micro_batch_size_per_gpu"] = args.per_device_train_batch_size

    if is_main_process:
        print("\n" + "="*60)
        print("On-Policy Distillation with vLLM (in-process)")
        print("="*60)
        print(f"Student Model: {args.model_path}")
        print(f"Teacher Model (vLLM): {args.teacher_model_path}")
        print(f"Data Path: {args.data_path}")
        print(f"Output Dir: {output_dir}")
        print(f"Lambda (on-policy prob): {args.lmbda}")
        print(f"Temperature: {args.temperature}")
        print(f"Max New Tokens: {args.max_new_tokens}")
        print(f"vLLM GPU Util: {args.vllm_gpu_memory_utilization}, TP: {args.vllm_tensor_parallel_size}")
        print(f"Distill Weight: {args.distill_weight}, Alpha: {args.distill_alpha}, Beta: {args.distill_beta}")
        print(f"Div: {args.div}, Phi Type: {args.phi_type}")
        print("="*60 + "\n")

    raw_dataset = load_dataset("json", data_files={"train": args.data_path}, split="train", streaming=True)

    def convert_to_prompt_completion(examples):
        prompts = []
        completions = []
        for i in range(len(examples['prompt'])):
            prompt = [
                {"role": "system", "content": "You are a helpful assistant. Please write a proper answer to the question."},
                {"role": "user", "content": examples['prompt'][i]}
            ]
            completion = [{"role": "assistant", "content": examples['completion'][i]}]
            prompts.append(prompt)
            completions.append(completion)
        return {"prompt": prompts, "completion": completions}

    def convert_to_text_format(examples):
        prompts = []
        completions = []
        system_prompt = "You are a helpful assistant. Please write a proper answer to the question."
        for i in range(len(examples['prompt'])):
            prompt = f"{system_prompt}\n\nQuestion: {examples['prompt'][i]}\n\nAnswer:"
            completion = f" {examples['completion'][i]}"
            prompts.append(prompt)
            completions.append(completion)
        return {"prompt": prompts, "completion": completions}

    def convert_message_to_prompt_completion(examples):
        prompts = []
        completions = []
        for i in range(len(examples['messages'])):
            query = examples['messages'][i][:-1]
            response = examples['messages'][i][-1]
            if query[0]["role"] == "system":
                prompt = query
            else:
                prompt = [{"role": "system", "content": "You are a helpful assistant."}]
                prompt.extend(query)
            completion = [response]
            prompts.append(prompt)
            completions.append(completion)
        return {"prompt": prompts, "completion": completions}

    def convert_messages_response_to_prompt_completion(examples):
        import json as json_mod
        all_conversations = []
        for i in range(len(examples['messages'])):
            raw_msgs = examples['messages'][i]
            if isinstance(raw_msgs, str):
                try:
                    msgs = json_mod.loads(raw_msgs)
                except (json_mod.JSONDecodeError, TypeError):
                    msgs = [{"role": "user", "content": raw_msgs}]
            elif isinstance(raw_msgs, list):
                msgs = raw_msgs
            else:
                msgs = [{"role": "user", "content": str(raw_msgs)}]
            if not msgs or msgs[0].get("role") != "system":
                msgs = [{"role": "system", "content": "You are a helpful assistant."}] + msgs
            response_text = examples['response'][i]
            full_conversation = msgs + [{"role": "assistant", "content": response_text}]
            all_conversations.append(full_conversation)
        return {"messages": all_conversations}

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.chat_template is None:
        tokenizer.chat_template = SIMPLE_CHAT_TEMPLATE

    teacher_model = args.teacher_model_path

    if args.max_steps is None and args.num_samples is not None:
        if isinstance(args.num_samples, str):
            args.num_samples = int(args.num_samples)
        effective_batch_size = args.per_device_train_batch_size * args.gradient_accumulation_steps * world_size
        args.max_steps = math.ceil((args.num_samples * args.num_train_epochs) / effective_batch_size)
        if is_main_process:
            print(f"Auto-calculated max_steps: {args.max_steps}")

    if args.max_steps is None:
        raise ValueError("max_steps must be specified for streaming datasets. Use --max_steps or --num_samples")

    model_path_lower = args.model_path.lower()
    if args.data_format == 'messages_response':
        convert_fn = convert_messages_response_to_prompt_completion
    elif args.data_format == 'messages':
        convert_fn = convert_message_to_prompt_completion
    elif args.data_format == 'prompt_completion':
        convert_fn = convert_to_prompt_completion
    else:
        if "instruct" in model_path_lower or "it" in model_path_lower:
            convert_fn = convert_to_prompt_completion
        else:
            convert_fn = convert_to_text_format

    dataset = raw_dataset.map(convert_fn, batched=True, remove_columns=raw_dataset.column_names)

    training_args = OnPolicyDistillConfig(
        output_dir=output_dir,
        logging_dir=logging_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        logging_strategy="steps",
        logging_steps=4,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=5,
        dataloader_drop_last=False,
        bf16=True,
        accelerator_config={"dispatch_batches": False},
        dataloader_num_workers=0,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
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
        alpha=args.alpha,
        gamma=args.gamma,
        distill_alpha=args.distill_alpha,
        distill_beta=args.distill_beta,
        distill_weight=args.distill_weight,
        phi_type=args.phi_type,
        div=args.div,
        deepspeed=deepspeed_config if args.use_deepspeed else None,
        lmbda=args.lmbda,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        top_p=args.top_p,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
        teacher_model_name_or_path=args.teacher_model_path,
        vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        vllm_tensor_parallel_size=args.vllm_tensor_parallel_size,
        vllm_dtype=args.vllm_dtype,
        vllm_max_model_len=args.vllm_max_model_len,
        vllm_device=args.vllm_device,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, dtype='auto', trust_remote_code=True, low_cpu_mem_usage=True,
    )

    trainer = OnPolicyDistillTrainer(
        model=model,
        teacher_model=teacher_model,
        train_dataset=dataset,
        args=training_args,
        peft_config=peft_config if args.use_lora else None,
        processing_class=tokenizer,
    )

    trainer.train()

    if is_main_process:
        print(f"\nTraining completed! Model saved to: {training_args.output_dir}")


if __name__ == "__main__":
    main()