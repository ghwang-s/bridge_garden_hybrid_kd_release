import os
import argparse
from pathlib import Path

os.environ['GLOO_SOCKET_IFNAME'] = 'eth0'
os.environ['NCCL_SOCKET_IFNAME'] = 'eth0'
os.environ['NCCL_IB_DISABLE'] = '1'
os.environ['NCCL_SOCKET_FAMILY'] = 'AF_INET'

os.environ['NCCL_NET'] = 'Socket'
os.environ['NCCL_P2P_LEVEL'] = 'NVL'  

from trl import SFTTrainer, SFTConfig
from trl.trainer.utils import SIMPLE_CHAT_TEMPLATE
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
            "device": "none",
            "pin_memory": True
        },
        "offload_param": {
            "device": "none",
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
    parser = argparse.ArgumentParser(description='SFT Training with DeepSpeed')
    
    parser.add_argument('--output_dir', type=str, default='./outputs/sft')
    parser.add_argument('--model_path', type=str, required=True, help='Student model path or Hugging Face model id')
    parser.add_argument('--teacher_model_path', type=str, default=None)
    parser.add_argument('--data_path', type=str, required=True, help='Training dataset path or dataset id')
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument('--per_device_train_batch_size', type=int, default=4)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=32)
    parser.add_argument('--num_train_epochs', type=int, default=2)
    parser.add_argument('--max_steps', type=int, default=None,
                       help='Maximum training steps (optional, will be calculated from num_samples if not provided)')
    parser.add_argument('--num_samples', type=int, default=None,
                       help='Total number of training samples (used to calculate max_steps for streaming datasets)')
    parser.add_argument('--max_length', type=int, default=2048)
    parser.add_argument('--use_lora', action='store_true', help='Enable LoRA')
    parser.add_argument('--lora_r', type=int, default=16)
    parser.add_argument('--lora_alpha', type=int, default=32)
    parser.add_argument('--seed', type=int, default=42)
    

    parser.add_argument('--alpha', type=float, default=0.05)
    parser.add_argument('--gamma', type=float, default=0.9999)
    parser.add_argument('--distill_alpha', type=float, default=1.0)
    parser.add_argument('--distill_beta', type=float, default=0.0)
    parser.add_argument('--distill_weight', type=float, default=0.2)
    parser.add_argument('--phi_type', type=str, default=None)
    parser.add_argument('--entropy_gamma', type=float, default=0.1)
    parser.add_argument('--curriculum_warmup_ratio', type=float, default=0.2)
    parser.add_argument('--div', type=str, default="ab")
    

    parser.add_argument('--data_format', type=str, default='auto',
                       choices=['auto', 'prompt_completion', 'messages_response', 'messages'],
                       help='Data format: auto (auto-detect), prompt_completion, messages_response, messages')
    
    # DeepSpeed
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
            print(f" Process group initialized with 1800s timeout")
    
    print(f" Distributed Training Environment:")
    print(f"   - RANK: {rank}, LOCAL_RANK: {local_rank}, WORLD_SIZE: {world_size}")
    print(f"   - Device: {device}")
    
    return local_rank, world_size, rank, device

def main():
    args = parse_args()
    
    local_rank, world_size, rank, device = init_distributed_training()
    is_main_process = local_rank == 0

    output_dir = f"{args.output_dir}/learning_rate_{args.learning_rate}"
    logging_dir = f"{output_dir}/logs"

    if is_main_process:
        ds_config_path = os.path.join(output_dir, "ds_config.json")
        if os.path.exists(ds_config_path):
            try:
                os.remove(ds_config_path)
            except OSError as e:
                print(f"Could not remove existing DeepSpeed config: {e}")
    # =======================================================
    
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
        print("Training Configuration:")
        print("="*60)
        print(f"Model Path: {args.model_path}")
        print(f"Teacher Model: {args.teacher_model_path or 'None'}")
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
        print(f"Gamma: {args.gamma}, Alpha: {args.alpha}, Distill Weight: {args.distill_weight}, Distill Alpha: {args.distill_alpha}, Distill Beta: {args.distill_beta}, Phi type: {args.phi_type}, DIV: {args.div}")
        
        if args.use_deepspeed:
            print(f"DeepSpeed Config: {config_name}")
        print("="*60 + "\n")

    

    if is_main_process:
        print(" Loading and processing dataset...")
    
    raw_dataset = load_dataset("json", data_files={"train": args.data_path}, split="train", streaming=True)
    
    def convert_to_prompt_completion(examples):
        prompts = []
        completions = []
        
        for i in range(len(examples['prompt'])):
            prompt = [
                {"role": "system", "content": "You are a helpful assistant. Please write a proper answer to the question."},
                {"role": "user", "content": examples['prompt'][i]}
            ]
            
            completion = [
                {"role": "assistant", "content": examples['completion'][i]}
            ]
            
            prompts.append(prompt)
            completions.append(completion)
        
        return {"prompt": prompts, "completion": completions}

    def convert_to_text_format(examples):
        prompts = []
        completions = []
        
        # system_prompt = "You are a helpful assistant. Please reason step by step to answer the question and provide the answer WITHIN \\boxed{}."
        system_prompt = "You are a helpful assistant. Please write a proper answer to the question."
                
        for i in range(len(examples['prompt'])):
            
            prompt = f"{system_prompt}\n\nQuestion: {examples['prompt'][i]}\n\nAnswer:"
            
            completion = f" {examples['completion'][i]}"
            
            prompts.append(prompt)
            completions.append(completion)
        
        return {
            "prompt": prompts,
            "completion": completions
        }

    def convert_message_to_prompt_completion(examples):
        prompts = []
        completions = []
        
        for i in range(len(examples['messages'])):
            query = examples['messages'][i][:-1]
            response = examples['messages'][i][-1]

            if query[0]["role"] == "system":
                prompt = query
            else:
                prompt = [
                    {"role": "system", "content": "You are a helpful assistant."},
                ]
                prompt.extend(query)
            
            completion = [response]
            
            prompts.append(prompt)
            completions.append(completion)
        
        return {"prompt": prompts, "completion": completions}

    def convert_messages_response_to_prompt_completion(examples):
        import json
        
        all_conversations = []
        
        for i in range(len(examples['messages'])):

            raw_msgs = examples['messages'][i]
            

            if isinstance(raw_msgs, str):
                try:
                    msgs = json.loads(raw_msgs)
                except (json.JSONDecodeError, TypeError):

                    msgs = [{"role": "user", "content": raw_msgs}]
            elif isinstance(raw_msgs, list):
                msgs = raw_msgs
            else:
                msgs = [{"role": "user", "content": str(raw_msgs)}]



            if not msgs or msgs[0].get("role") != "system":
                msgs = [
                    {"role": "system", "content": "You are a helpful assistant."}
                ] + msgs
            

            response_text = examples['response'][i]
            


            full_conversation = msgs + [
                {"role": "assistant", "content": response_text}
            ]
            
            all_conversations.append(full_conversation)
        

        return {"messages": all_conversations}

    if is_main_process:
        print(" Loading and processing dataset (Main process starting first)...")


    

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
    

    if tokenizer.chat_template is None:
        tokenizer.chat_template = SIMPLE_CHAT_TEMPLATE
        if is_main_process:
            print(f" Chat template set to SIMPLE_CHAT_TEMPLATE")
    

    teacher_model = None
    if args.teacher_model_path and args.distill_weight > 0:
        if is_main_process:
            output_dir = f"{args.output_dir}/learning_rate_{args.learning_rate}/distill_weight_{args.distill_weight}_distill_alpha_{args.distill_alpha}_distill_beta_{args.distill_beta}"
            if args.phi_type == "entropy_sft_distill":
                output_dir = f"{output_dir}_ent_gamma_{args.entropy_gamma}"
            if args.phi_type == "curriculum_weight":
                output_dir = f"{output_dir}_cur_warmup_{args.curriculum_warmup_ratio}"
            logging_dir = f"{output_dir}/logs"
        
        teacher_model = args.teacher_model_path
    
    if is_main_process:
        print("Creating SFTTrainer...")


    if args.max_steps is None and args.num_samples is not None:

        if isinstance(args.num_samples, str):
            args.num_samples = int(args.num_samples)
            if is_main_process:
                print(f"  Converted num_samples from string to int: {args.num_samples}")


        effective_batch_size = args.per_device_train_batch_size * args.gradient_accumulation_steps * world_size
        args.max_steps = math.ceil((args.num_samples * args.num_train_epochs) / effective_batch_size)

        if is_main_process:
            print("\n" + "="*60)
            print(" Auto-calculated Training Steps from num_samples:")
            print("="*60)
            print(f"Total samples: {args.num_samples}")
            print(f"Num epochs: {args.num_train_epochs}")
            print(f"Per-device batch size: {args.per_device_train_batch_size}")
            print(f"Gradient accumulation steps: {args.gradient_accumulation_steps}")
            print(f"World size: {world_size}")
            print(f"Effective batch size: {effective_batch_size}")
            print(f"Calculated max_steps: {args.max_steps}")
            print("="*60 + "\n")


    if args.max_steps is None:
        if is_main_process:
            print("\n" + ""*30)
            print("ERROR: Streaming dataset requires max_steps to be specified!")
            print("="*60)
            print("You have 2 options:")
            print()
            print("Option 1: Specify max_steps directly")
            print("  python train.py --max_steps 10000 ...")
            print()
            print("Option 2: Specify num_samples and num_train_epochs (auto-calculate)")
            print("  python train.py --num_samples 2027181 --num_train_epochs 2 ...")
            print()
            print("Current values:")
            print(f"  - max_steps: {args.max_steps}")
            print(f"  - num_samples: {args.num_samples} (type: {type(args.num_samples).__name__})")
            print(f"  - num_train_epochs: {args.num_train_epochs}")
            print("="*60 + "\n")
        raise ValueError("max_steps must be specified for streaming datasets. Use --max_steps or --num_samples")

    if is_main_process:
        print(f" Training will run for {args.max_steps} steps\n")


    final_num_train_epochs = args.num_train_epochs
    final_max_steps = args.max_steps

    if is_main_process:
        print(f"   final_num_train_epochs = {final_num_train_epochs}, type = {type(final_num_train_epochs)}")
        print(f"   final_max_steps = {final_max_steps}, type = {type(final_max_steps)}")

    training_args = SFTConfig(
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
        num_train_epochs=final_num_train_epochs,
        max_steps=final_max_steps,
        lr_scheduler_type="cosine",
        learning_rate=args.learning_rate,
        warmup_ratio=0.03,
        seed=args.seed,
        optim="adamw_torch",
        ddp_timeout=10800,
        # bf16=False,
        # fp16=False,

        report_to='tensorboard' if is_main_process else None,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        ddp_find_unused_parameters=False,
        max_length=args.max_length,
        

        # save_safetensors=False,

        alpha=args.alpha,
        gamma=args.gamma,
        distill_alpha=args.distill_alpha,
        distill_beta=args.distill_beta,
        distill_weight=args.distill_weight,
        phi_type=args.phi_type,
        entropy_gamma=args.entropy_gamma,
        curriculum_warmup_ratio=args.curriculum_warmup_ratio,
        div=args.div,


        deepspeed=deepspeed_config if args.use_deepspeed else None,
    )


    model_path_lower = args.model_path.lower()
    

    if args.data_format == 'messages_response':

        convert_fn = convert_messages_response_to_prompt_completion
        convert_desc = "Converting messages(JSON) + response to prompt-completion format"
        if is_main_process:
            print(" Using messages_response format (JSON string messages + response)")
    elif args.data_format == 'messages':

        convert_fn = convert_message_to_prompt_completion
        convert_desc = "Converting messages list to prompt-completion format"
        if is_main_process:
            print(" Using messages format (messages list)")
    elif args.data_format == 'prompt_completion':

        convert_fn = convert_to_prompt_completion
        convert_desc = "Converting prompt + completion to prompt-completion format"
        if is_main_process:
            print(" Using prompt_completion format")
    else:  # auto

        if "instruct" in model_path_lower or "it" in model_path_lower:

            convert_fn = convert_to_prompt_completion
            convert_desc = "Converting messages to prompt-completion format (Instruct)"
            if is_main_process:
                print(" Detected Instruct model, using conversational format")
        else:

            convert_fn = convert_to_text_format
            convert_desc = "Converting to text format (Base model)"
            if is_main_process:
                print(" Detected Base model, using text format")
    


    dataset = raw_dataset.map(
        convert_fn,
        batched=True,
        remove_columns=raw_dataset.column_names
    )

    if is_main_process:
        print(f" Dataset processing complete (streaming mode)")

    if is_main_process:
        print(f" Dataset loaded in streaming mode (no disk caching)")
        print(f" Effective batch size: {args.per_device_train_batch_size * args.gradient_accumulation_steps * world_size}")
        print(f" Training will run for {args.max_steps} steps")
        print(f"\nModel output_dir: {training_args.output_dir}")
        print(f"Logging dir: {training_args.logging_dir}")
        if args.use_deepspeed:
            print(f"DeepSpeed config file: {deepspeed_config}")
        print()

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        dtype='auto',
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    

    trainer = SFTTrainer(
        model=model,
        teacher_model=teacher_model,
        train_dataset=dataset,
        args=training_args,
        peft_config=peft_config if args.use_lora else None,
        processing_class=tokenizer
    )
    
    trainer.train()
    

    if is_main_process:
        print(f"\nSaving final model to {training_args.output_dir}...")
    
    if is_main_process:
        print(f"\nTraining completed!")
        print(f"Model saved to: {training_args.output_dir}")


if __name__ == "__main__":
    main()
