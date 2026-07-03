from dataclasses import dataclass, field
from typing import Any, Optional

from .sft_config import SFTConfig


@dataclass
class OnPolicyDistillConfig(SFTConfig):

    lmbda: float = field(default=1.0, metadata={"help": "Probability of using on-policy student generations (0.0=off-policy only, 1.0=always on-policy)"})
    temperature: float = field(default=1.0, metadata={"help": "Sampling temperature for on-policy generation"})
    max_new_tokens: int = field(default=512, metadata={"help": "Maximum number of new tokens to generate for on-policy completions"})
    top_p: float = field(default=1.0, metadata={"help": "Top-p sampling parameter for generation"})
    top_k: int = field(default=-1, metadata={"help": "Top-k sampling parameter for generation (-1 for no top-k)"})
    repetition_penalty: float = field(default=1.0, metadata={"help": "Repetition penalty for generation"})
    seq_kd: bool = field(default=False, metadata={"help": "If True, generate from teacher model instead of student for sequence-level KD"})

    teacher_model_name_or_path: Optional[str] = field(default=None, metadata={"help": "Path to the teacher model for vLLM inference"})
    vllm_gpu_memory_utilization: float = field(default=0.5, metadata={"help": "GPU memory utilization for vLLM teacher model"})
    vllm_tensor_parallel_size: int = field(default=1, metadata={"help": "Tensor parallel size for vLLM teacher model"})
    vllm_dtype: str = field(default="auto", metadata={"help": "Data type for vLLM teacher model (auto, float16, bfloat16)"})
    vllm_max_model_len: Optional[int] = field(default=None, metadata={"help": "Max model length for vLLM (defaults to max_length + max_new_tokens)"})
    vllm_device: Optional[str] = field(default=None, metadata={"help": "Device for vLLM teacher model, e.g. 'cuda:1'. If None, auto-select."})

    disable_dropout: bool = field(default=True, metadata={"help": "Disable dropout during training"})

    def __post_init__(self):
        super().__post_init__()
        if self.lmbda < 0.0 or self.lmbda > 1.0:
            raise ValueError("lmbda must be in the range [0.0, 1.0].")