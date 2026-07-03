import random
import logging
import os
from typing import Any, Callable, Optional, Union

import torch
import torch.nn as nn
from transformers import (
    EvalPrediction,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    ProcessorMixin,
)
from transformers.utils import is_peft_available

from .on_policy_distill_config import OnPolicyDistillConfig
from .sft_trainer import SFTTrainer
from ..import_utils import is_vllm_available

if is_peft_available():
    from peft import PeftConfig

logger = logging.getLogger(__name__)


class OnPolicyDistillTrainer(SFTTrainer):

    _tag_names = ["trl", "on-policy-distill"]

    def __init__(
        self,
        model: Optional[Union[str, PreTrainedModel]] = None,
        teacher_model: Optional[Union[str, PreTrainedModel]] = None,
        args: Optional[OnPolicyDistillConfig] = None,
        data_collator: Optional[Any] = None,
        train_dataset: Optional[Any] = None,
        eval_dataset: Optional[Any] = None,
        processing_class: Optional[Union[PreTrainedTokenizerBase, ProcessorMixin]] = None,
        compute_loss_func: Optional[Callable] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], dict]] = None,
        callbacks: Optional[list] = None,
        optimizers: tuple = (None, None),
        optimizer_cls_and_kwargs: Optional[tuple] = None,
        preprocess_logits_for_metrics: Optional[Callable] = None,
        peft_config: Optional["PeftConfig"] = None,
        formatting_func: Optional[Callable] = None,
    ):
        if args is None:
            args = OnPolicyDistillConfig("on-policy-distill")

        if args.disable_dropout:
            if isinstance(model, PreTrainedModel):
                for module in model.modules():
                    if isinstance(module, nn.Dropout):
                        module.p = 0.0

        self.lmbda = args.lmbda

        if isinstance(teacher_model, str):
            self._teacher_model_path = teacher_model
        elif args.teacher_model_name_or_path is not None:
            self._teacher_model_path = args.teacher_model_name_or_path
        else:
            self._teacher_model_path = None

        super().__init__(
            model=model,
            teacher_model=teacher_model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            compute_loss_func=compute_loss_func,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            optimizer_cls_and_kwargs=optimizer_cls_and_kwargs,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
            peft_config=peft_config,
            formatting_func=formatting_func,
        )

        self._init_vllm_teacher()

    def _init_vllm_teacher(self):
        if self._teacher_model_path is None:
            logger.warning("No teacher model path provided, on-policy generation disabled.")
            self.vllm_teacher = None
            return

        if not is_vllm_available():
            raise ImportError(
                "vLLM is not available. Please install vLLM with `pip install trl[vllm]`."
            )

        if not self.accelerator.is_main_process:
            self.vllm_teacher = None
            self.accelerator.wait_for_everyone()
            return

        from vllm import LLM

        max_model_len = self.args.vllm_max_model_len
        if max_model_len is None:
            max_model_len = (self.args.max_length or 1024) + self.args.max_new_tokens

        env_backup = os.environ.get("CUDA_VISIBLE_DEVICES", None)

        vllm_device = self.args.vllm_device
        if vllm_device is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = vllm_device.replace("cuda:", "")

        logger.info(f"Initializing vLLM teacher model from {self._teacher_model_path}")
        try:
            self.vllm_teacher = LLM(
                model=self._teacher_model_path,
                tensor_parallel_size=self.args.vllm_tensor_parallel_size,
                gpu_memory_utilization=self.args.vllm_gpu_memory_utilization,
                max_model_len=max_model_len,
                dtype=self.args.vllm_dtype,
                trust_remote_code=True,
            )
        finally:
            if env_backup is not None:
                os.environ["CUDA_VISIBLE_DEVICES"] = env_backup
            elif "CUDA_VISIBLE_DEVICES" in os.environ and vllm_device is not None:
                del os.environ["CUDA_VISIBLE_DEVICES"]

        logger.info("vLLM teacher model initialized successfully")
        self.accelerator.wait_for_everyone()

    def _generate_with_vllm_teacher(self, prompts_text):
        from vllm import SamplingParams

        is_distributed = self.accelerator.num_processes > 1

        if is_distributed:
            from accelerate.utils import gather_object
            from torch.distributed import broadcast_object_list
            all_prompts_text = gather_object(prompts_text)
        else:
            all_prompts_text = prompts_text

        if self.accelerator.is_main_process:
            sampling_params = SamplingParams(
                n=1,
                temperature=self.args.temperature,
                top_p=self.args.top_p,
                top_k=self.args.top_k if self.args.top_k > 0 else -1,
                max_tokens=self.args.max_new_tokens,
                repetition_penalty=self.args.repetition_penalty,
            )

            outputs = self.vllm_teacher.generate(
                all_prompts_text,
                sampling_params=sampling_params,
                use_tqdm=False,
            )

            all_completion_ids = []
            for output in outputs:
                token_ids = list(output.outputs[0].token_ids)
                all_completion_ids.append(token_ids)
        else:
            all_completion_ids = None

        if is_distributed:
            obj_list = [all_completion_ids]
            broadcast_object_list(obj_list, from_process=0)
            all_completion_ids = obj_list[0]

            process_slice = slice(
                self.accelerator.process_index * len(prompts_text),
                (self.accelerator.process_index + 1) * len(prompts_text),
            )
            return all_completion_ids[process_slice]
        else:
            return all_completion_ids

    def _build_inputs_from_completions(self, prompts_text, completion_ids_list):
        tokenizer = self.processing_class
        pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

        all_input_ids = []
        all_labels = []
        max_len = 0

        for prompt_text, comp_ids in zip(prompts_text, completion_ids_list):
            prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
            if self.args.max_length is not None:
                max_completion_len = self.args.max_length - len(prompt_ids)
                if max_completion_len <= 0:
                    prompt_ids = prompt_ids[:self.args.max_length // 2]
                    max_completion_len = self.args.max_length - len(prompt_ids)
                comp_ids = comp_ids[:max_completion_len]

            full_ids = prompt_ids + list(comp_ids)
            labels = [-100] * len(prompt_ids) + list(comp_ids)

            all_input_ids.append(full_ids)
            all_labels.append(labels)
            max_len = max(max_len, len(full_ids))

        if self.args.max_length is not None:
            max_len = min(max_len, self.args.max_length)

        batch_input_ids = []
        batch_attention_mask = []
        batch_labels = []

        for input_ids, labels in zip(all_input_ids, all_labels):
            input_ids = input_ids[:max_len]
            labels = labels[:max_len]

            pad_len = max_len - len(input_ids)
            batch_input_ids.append(input_ids + [pad_token_id] * pad_len)
            batch_attention_mask.append([1] * len(input_ids) + [0] * pad_len)
            batch_labels.append(labels + [-100] * pad_len)

        device = self.accelerator.device
        return {
            "input_ids": torch.tensor(batch_input_ids, dtype=torch.long, device=device),
            "attention_mask": torch.tensor(batch_attention_mask, dtype=torch.long, device=device),
            "labels": torch.tensor(batch_labels, dtype=torch.long, device=device),
        }

    def _extract_prompts_from_inputs(self, inputs):
        tokenizer = self.processing_class
        input_ids = inputs["input_ids"]
        labels = inputs.get("labels", None)

        prompts_text = []
        for i in range(input_ids.size(0)):
            ids = input_ids[i].tolist()

            if labels is not None:
                lab = labels[i].tolist()
                prompt_len = 0
                for j, l in enumerate(lab):
                    if l != -100:
                        prompt_len = j
                        break
                else:
                    prompt_len = len(ids)
                prompt_ids = ids[:prompt_len]
            elif "completion_mask" in inputs:
                mask = inputs["completion_mask"][i].tolist()
                prompt_len = 0
                for j, m in enumerate(mask):
                    if m == 1:
                        prompt_len = j
                        break
                else:
                    prompt_len = len(ids)
                prompt_ids = ids[:prompt_len]
            else:
                prompt_ids = ids

            pad_token_id = tokenizer.pad_token_id
            if pad_token_id is not None:
                prompt_ids = [t for t in prompt_ids if t != pad_token_id]

            prompt_text = tokenizer.decode(prompt_ids, skip_special_tokens=False)
            prompts_text.append(prompt_text)

        return prompts_text

    def training_step(
        self, model: nn.Module, inputs: dict[str, Union[torch.Tensor, Any]], num_items_in_batch: Optional[int] = None
    ) -> torch.Tensor:
        is_packed = "seq_lengths" in inputs
        has_vllm = self.vllm_teacher is not None or not self.accelerator.is_main_process
        can_do_on_policy = has_vllm and not is_packed

        if can_do_on_policy:
            do_on_policy = random.random() < self.lmbda
            if self.accelerator.num_processes > 1:
                flag_tensor = torch.tensor([1 if do_on_policy else 0], device=self.accelerator.device)
                torch.distributed.broadcast(flag_tensor, src=0)
                do_on_policy = flag_tensor.item() == 1

            if do_on_policy:
                prompts_text = self._extract_prompts_from_inputs(inputs)
                completion_ids_list = self._generate_with_vllm_teacher(prompts_text)
                new_inputs = self._build_inputs_from_completions(prompts_text, completion_ids_list)
                inputs["input_ids"] = new_inputs["input_ids"]
                inputs["attention_mask"] = new_inputs["attention_mask"]
                inputs["labels"] = new_inputs["labels"]
                if "completion_mask" in inputs:
                    del inputs["completion_mask"]
                if "seq_lengths" in inputs:
                    del inputs["seq_lengths"]

        loss = super().training_step(model, inputs, num_items_in_batch)
        return loss
