# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import textwrap
import warnings
from typing import Any, Callable, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    BaseImageProcessor,
    DataCollator,
    FeatureExtractionMixin,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    ProcessorMixin,
)
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import EvalPrediction
from transformers.utils import is_liger_kernel_available, is_peft_available

from ..models import prepare_deepspeed
from ..models.utils import unwrap_model_for_generation
from .gkd_config import GKDConfig
from .sft_trainer import SFTTrainer
from .utils import DataCollatorForChatML, disable_dropout_in_model, empty_cache


if is_peft_available():
    from peft import PeftConfig

if is_liger_kernel_available():
    from liger_kernel.chunked_loss import LigerFusedLinearJSDLoss


class GKDTrainer(SFTTrainer):
    """Trainer for Generalized Knowledge Distillation (GKD) of language models.

    For details on GKD, see the paper: [On-Policy Distillation of Language Models: Learning from Self-Generated
    Mistakes](https://huggingface.co/papers/2306.13649).

    Args:
        model ([`~transformers.PreTrainedModel`] or `torch.nn.Module` or `str`, *optional*):
            Model to be trained, or the string identifier of the model to be instantiated from a pretrained model.
        teacher_model ([`~transformers.PreTrainedModel`] or `torch.nn.Module` or `str`, *optional*):
            Teacher model for knowledge distillation, or the string identifier of the model to be instantiated from a
            pretrained model.
        args ([`GKDConfig`], *optional*):
            Training arguments.
        data_collator ([`~transformers.DataCollator`], *optional*):
            Data collator to batch samples from the dataset. It defaults to a [`DataCollatorForChatML`] using the
            `processing_class`.
        train_dataset ([`~datasets.Dataset`], *optional*):
            Dataset for training.
        eval_dataset ([`~datasets.Dataset`] or `dict` of [`~datasets.Dataset`], *optional*):
            Dataset for evaluation.
        processing_class ([`~transformers.PreTrainedTokenizerBase`], [`~transformers.BaseImageProcessor`], [`~transformers.FeatureExtractionMixin`] or [`~transformers.ProcessorMixin`], *optional*):
           Class to process the data.
        compute_metrics (`Callable`, *optional*):
            Function to compute metrics at evaluation. Must take in an [`~transformers.EvalPrediction`] and return a
            dictionary string to float.
        callbacks (`list` of [`~transformers.TrainerCallback`], *optional*):
            Callbacks to use during training.
        optimizers (`tuple` of `torch.optim.Optimizer` and `torch.optim.lr_scheduler.LambdaLR`, *optional*, defaults to `(None, None)`):
            Tuple containing the optimizer and the learning rate scheduler to use for training.
        preprocess_logits_for_metrics (`Callable`, *optional*):
            Function to preprocess the logits before computing the metrics. Must take in the `logits` and `labels` and
            return the logits to be used for metrics computation.
        peft_config ([`~peft.PeftConfig`], *optional*):
            PEFT configuration to use PEFT for training. If `None`, PEFT is not used. If provided, the `model` will be
            wrapped with the specified PEFT adapter.
        formatting_func (`Callable`, *optional*):
            Function to format the dataset. Must take in an example and return an example.
    """

    _tag_names = ["trl", "gkd"]
    _name = "GKD"
    _paper = {
        "title": "On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes",
        "id": "2306.13649",
        # docstyle-ignore
        "citation": textwrap.dedent("""\
            @inproceedings{agarwal2024on-policy,
                title        = {{On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes}},
                author       = {Rishabh Agarwal and Nino Vieillard and Yongchao Zhou and Piotr Stanczyk and Sabela Ramos Garea and Matthieu Geist and Olivier Bachem},
                year         = 2024,
                booktitle    = {The Twelfth International Conference on Learning Representations, {ICLR} 2024, Vienna, Austria, May 7-11, 2024},
                publisher    = {OpenReview.net},
                url          = {https://openreview.net/forum?id=3zKtaqxLhW},
            }"""),
    }

    @staticmethod
    def _is_deepspeed_engine(model: nn.Module) -> bool:
        if getattr(model, "_trl_is_deepspeed_inference", False):
            return True
        module_name = model.__class__.__module__
        class_name = model.__class__.__name__
        return module_name.startswith("deepspeed") or class_name in {"DeepSpeedEngine", "InferenceEngine"}

    def __init__(
        self,
        model: Optional[Union[PreTrainedModel, nn.Module, str]] = None,
        teacher_model: Union[PreTrainedModel, nn.Module, str] = None,
        args: Optional[GKDConfig] = None,
        data_collator: Optional[DataCollator] = None,  # type: ignore
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Union[Dataset, dict[str, Dataset]]] = None,
        processing_class: Optional[
            Union[PreTrainedTokenizerBase, BaseImageProcessor, FeatureExtractionMixin, ProcessorMixin]
        ] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], dict]] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        peft_config: Optional["PeftConfig"] = None,
        formatting_func: Optional[Callable] = None,
    ):
        if not os.environ.get("TRL_EXPERIMENTAL_SILENCE"):
            warnings.warn(
                "This trainer will soon be moved to trl.experimental and is a candidate for removal. If you rely on "
                "it and want it to remain, please share your comments here: "
                "https://github.com/huggingface/trl/issues/4223. Silence this warning by setting environment variable "
                "TRL_EXPERIMENTAL_SILENCE=1."
            )
        # Ensure Trainer does not drop non-signature columns used by the collator (e.g., "prompts")
        args.remove_unused_columns = False
        # Respect a user-provided data_collator; otherwise, provide a ChatML collator that
        if data_collator is None:
            data_collator = DataCollatorForChatML(tokenizer=processing_class, max_length=args.max_length)

        # Ensure SFTTrainer does not pre-process the dataset when using a ChatML collator,
        # so that raw conversational fields (e.g., "messages") remain available to the collator.
        if args.dataset_kwargs is None:
            args.dataset_kwargs = {"skip_prepare_dataset": True}
        else:
            args.dataset_kwargs["skip_prepare_dataset"] = True

        # Liger fused GKD loss (JSD)
        self.use_liger_gkd_loss = False
        if args.use_liger_kernel:
            self.liger_jsd_loss = LigerFusedLinearJSDLoss(
                beta=args.beta,
                ignore_index=-100,
                temperature=args.temperature,
                compiled=False,
            )
            self.use_liger_gkd_loss = True

        super().__init__(
            model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
            peft_config=peft_config,
            formatting_func=formatting_func,
        )

        if args.teacher_model_init_kwargs is None:
            teacher_model_init_kwargs = {}
        elif not isinstance(teacher_model, str):
            raise ValueError(
                "You passed teacher_model_init_kwargs to the GKDConfig, but your teacher_model is already instantiated."
            )
        else:
            teacher_model_init_kwargs = args.teacher_model_init_kwargs
            teacher_model_init_kwargs["dtype"] = (
                teacher_model_init_kwargs["dtype"]
                if teacher_model_init_kwargs["dtype"] in ["auto", None]
                else getattr(torch, teacher_model_init_kwargs["dtype"])
            )

        if isinstance(teacher_model, str):
            teacher_model = AutoModelForCausalLM.from_pretrained(teacher_model, **teacher_model_init_kwargs)

        # Disable dropout in the model
        if args.disable_dropout:
            disable_dropout_in_model(self.model)

        teacher_already_wrapped = self._is_deepspeed_engine(teacher_model)

        if self.is_deepspeed_enabled:
            if teacher_already_wrapped:
                self.teacher_model = teacher_model
                self.teacher_model.eval()
            else:
                self.teacher_model = prepare_deepspeed(teacher_model, self.accelerator)
        elif teacher_already_wrapped:
            self.teacher_model = teacher_model
            self.teacher_model.eval()
        else:
            self.teacher_model = self.accelerator.prepare_model(teacher_model, evaluation_mode=True)

        self.lmbda = args.lmbda
        self.beta = args.beta
        self.temperature = args.temperature
        self.seq_kd = args.seq_kd

        self.generation_config = GenerationConfig(
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            do_sample=True,
            top_k=0,
            use_cache=False if args.gradient_checkpointing else True,
            pad_token_id=self.processing_class.pad_token_id,
        )
        # Set custom EOS tokens if they are specified by the model's generation
        # config. This is important for models with the Llama 3 chat template,
        # which use special tokens <|eot_id|> and <|eom_id|> to mark the end of
        # turns or messages.
        if (
            hasattr(self.model.generation_config, "eos_token_id")
            and self.model.generation_config.eos_token_id is not None
        ):
            self.generation_config.eos_token_id = self.model.generation_config.eos_token_id

    @staticmethod
    def generalized_jsd_loss(
        student_logits, teacher_logits, labels=None, beta=0.5, temperature=1.0, reduction="batchmean"
    ):
        """
        Compute the generalized Jensen-Shannon Divergence loss for knowledge distillation using F.kl_div. See Eq. (1)
        of https://huggingface.co/papers/2306.13649 for the definition.

        Args:
            student_logits:
                Tensor of shape (batch_size, sequence_length, vocab_size)
            teacher_logits:
                Tensor of shape (batch_size, sequence_length, vocab_size)
            labels:
                Tensor of shape (batch_size, sequence_length) with -100 for padding tokens to ignore when computing
                loss
            beta:
                Interpolation coefficient between 0 and 1 (default: 0.5)
            temperature:
                Softmax temperature (default: 1.0)
            reduction:
                Specifies the reduction to apply to the output (default: 'batchmean')

        Returns:
            loss: Scalar tensor with the generalized JSD loss
        """

        # Apply temperature scaling
        student_logits = student_logits / temperature
        teacher_logits = teacher_logits / temperature

        # Compute log probabilities for student and probabilities for teacher
        student_log_probs = F.log_softmax(student_logits, dim=-1)
        teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)

        if beta == 0:
            jsd = F.kl_div(student_log_probs, teacher_log_probs, reduction="none", log_target=True)
        elif beta == 1:
            jsd = F.kl_div(teacher_log_probs, student_log_probs, reduction="none", log_target=True)
        else:
            # Compute the log of the mixture distribution
            # log(a + b) = log(exp(log(a)) + exp(log(b))) -> for mixture
            beta = torch.tensor(beta, dtype=student_log_probs.dtype)
            mixture_log_probs = torch.logsumexp(
                torch.stack([student_log_probs + torch.log(1 - beta), teacher_log_probs + torch.log(beta)]),
                dim=0,
            )

            # Compute KL divergences using F.kl_div
            # PyTorch differs from the standard mathematical definition, so the order of the probability distributions is swapped compared to that defined in the paper.
            kl_teacher = F.kl_div(mixture_log_probs, teacher_log_probs, reduction="none", log_target=True)
            kl_student = F.kl_div(mixture_log_probs, student_log_probs, reduction="none", log_target=True)

            # Compute the Generalized Jensen-Shannon Divergence
            jsd = beta * kl_teacher + (1 - beta) * kl_student

        # Masking
        if labels is not None:
            mask = labels != -100
            jsd = jsd[mask]

        # Apply reduction
        if reduction == "batchmean":
            return jsd.sum() / mask.sum() if labels is not None else jsd.sum() / jsd.size(0)
        elif reduction == "sum":
            return jsd.sum()
        elif reduction == "mean":
            return jsd.mean()
        else:
            return jsd

    @staticmethod
    def ab_div_loss(
        student_logits,
        teacher_logits,
        labels,
        alpha=1.0,
        beta=0.0,
        gamma=1.0,
        warmup=0.0,
        current_step=0,
        total_steps=0,
        weight=None,
    ):

        if student_logits.size(-1) != teacher_logits.size(-1):
            common_vocab_size = min(student_logits.size(-1), teacher_logits.size(-1))
            student_logits = student_logits[..., :common_vocab_size]
            teacher_logits = teacher_logits[..., :common_vocab_size]

        eps = 1e-8
        log_p = F.log_softmax(teacher_logits, dim=-1, dtype=torch.float32)
        log_q = F.log_softmax(student_logits, dim=-1, dtype=torch.float32)

        if abs(alpha) < eps and abs(beta) < eps:
            divergence = 0.5 * torch.sum((log_q - log_p).pow(2), dim=-1)
        elif abs(alpha) < eps:
            safe_ratio_beta = torch.where(torch.isfinite(log_q - log_p), log_q - log_p, torch.zeros_like(log_q - log_p))
            divergence = torch.sum(
                torch.exp(beta * log_q) * (beta * safe_ratio_beta - 1) + torch.exp(beta * log_p),
                dim=-1,
            ) / (beta**2)
        elif abs(beta) < eps:
            safe_ratio_alpha = torch.where(torch.isfinite(log_p - log_q), log_p - log_q, torch.zeros_like(log_p - log_q))
            divergence = torch.sum(
                torch.exp(alpha * log_p) * (alpha * safe_ratio_alpha - 1) + torch.exp(alpha * log_q),
                dim=-1,
            ) / (alpha**2)
        elif abs(alpha + beta) < eps:
            safe_log_r = torch.where(torch.isfinite(log_q - log_p), log_q - log_p, torch.zeros_like(log_q - log_p))
            divergence = torch.sum(alpha * safe_log_r + torch.exp(-alpha * safe_log_r) - 1, dim=-1) / (alpha**2)
        else:
            apb = alpha + beta
            term1 = torch.exp(torch.logsumexp(alpha * log_p + beta * log_q, dim=-1))
            term2 = (alpha / apb) * torch.exp(torch.logsumexp(apb * log_p, dim=-1))
            term3 = (beta / apb) * torch.exp(torch.logsumexp(apb * log_q, dim=-1))
            divergence = -(term1 - term2 - term3) / (alpha * beta)

        mask = (labels != -100).float()

        if abs(gamma - 1.0) < eps:
            decay_weights = mask
        else:
            token_indices = torch.cumsum(mask, dim=-1) - 1
            if warmup > 0 and total_steps > 0:
                gamma_ = gamma + (1 - gamma) * min((current_step / (total_steps * warmup)), 1)
            else:
                gamma_ = gamma
            decay_weights = (gamma_**token_indices) * mask

        safe_divergence = torch.where(torch.isfinite(divergence), divergence, torch.zeros_like(divergence))
        token_losses = safe_divergence * decay_weights

        if weight is not None:
            if weight.dim() > 1:
                weight = weight.squeeze()
            weighted_sum = (weight * token_losses).sum()
            sum_of_weights = torch.clamp(mask.sum(), min=eps)
            loss = weighted_sum / sum_of_weights
        else:
            loss = token_losses.sum() / torch.clamp(mask.sum(), min=1.0)

        return loss

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if self.use_liger_gkd_loss:
            # Forward only through the base models (avoid lm_head to save memory)
            unwrapped_student = self.accelerator.unwrap_model(model)
            if hasattr(unwrapped_student, "get_decoder") and unwrapped_student.get_decoder() is not None:
                base_student = unwrapped_student.get_decoder()
            else:
                base_student = getattr(
                    unwrapped_student, getattr(unwrapped_student, "base_model_prefix", "model"), unwrapped_student
                )

            student_outputs = base_student(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                output_hidden_states=True,
                use_cache=False,
            )

            self.teacher_model.eval()
            unwrapped_teacher = self.accelerator.unwrap_model(self.teacher_model)
            if hasattr(unwrapped_teacher, "get_decoder") and unwrapped_teacher.get_decoder() is not None:
                base_teacher = unwrapped_teacher.get_decoder()
            else:
                base_teacher = getattr(
                    unwrapped_teacher, getattr(unwrapped_teacher, "base_model_prefix", "model"), unwrapped_teacher
                )
            with torch.no_grad():
                teacher_outputs = base_teacher(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    output_hidden_states=True,
                    use_cache=False,
                )

            student_hidden = student_outputs.last_hidden_state[:, :-1].contiguous()
            teacher_hidden = teacher_outputs.last_hidden_state[:, :-1].contiguous()

            labels_mask = inputs["labels"] != -100
            masked_input_ids = torch.where(
                labels_mask, inputs["input_ids"], torch.full_like(inputs["input_ids"], -100)
            )
            true_labels = masked_input_ids[:, 1:].contiguous()

            student_head = unwrapped_student.get_output_embeddings()
            teacher_head = unwrapped_teacher.get_output_embeddings()

            loss = self.liger_jsd_loss(
                student_input=student_hidden,
                student_weight=student_head.weight,
                teacher_input=teacher_hidden,
                teacher_weight=teacher_head.weight,
                true_labels=true_labels,
                student_bias=getattr(student_head, "bias", None),
                teacher_bias=getattr(teacher_head, "bias", None),
            )
        else:
            student_outputs = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
            )

            self.teacher_model.eval()
            with torch.no_grad():
                teacher_outputs = self.teacher_model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                )

            prompt_lengths = inputs["prompts"].shape[1]
            shifted_student_logits = student_outputs.logits[:, prompt_lengths - 1 : -1, :]
            shifted_teacher_logits = teacher_outputs.logits[:, prompt_lengths - 1 : -1, :]
            shifted_labels = inputs["labels"][:, prompt_lengths:]

            kd_loss_type = "ab_div"
            if kd_loss_type == "ab_div":
                alpha = getattr(self.args, "distill_alpha", 1.0)
                beta = getattr(self.args, "distill_beta", 0.0)
                gamma = getattr(self.args, "ab_gamma", 1.0)
                warmup = getattr(self.args, "ab_warmup", 0.0)
                current_step = getattr(getattr(self, "state", None), "global_step", 0) or 0
                total_steps = getattr(self.args, "max_steps", 0)
                kd_loss = self.ab_div_loss(
                    shifted_student_logits,
                    shifted_teacher_logits,
                    shifted_labels,
                    alpha=alpha,
                    beta=beta,
                    gamma=gamma,
                    warmup=warmup,
                    current_step=current_step,
                    total_steps=total_steps,
                    weight=None,
                )
            else:
                kd_loss = self.generalized_jsd_loss(
                    student_logits=shifted_student_logits,
                    teacher_logits=shifted_teacher_logits,
                    labels=shifted_labels,
                    beta=self.beta,
                )

            # Add CE loss and return weighted sum with KD loss
            ce_weight = 0.0
            kd_weight = 1.0
            if self.args.phi_type == "sft":
                attn_valid = inputs["attention_mask"][:, prompt_lengths:]
                ce_logits_flat = shifted_student_logits.reshape(-1, shifted_student_logits.size(-1))
                ce_labels_flat = shifted_labels.reshape(-1)
                ce_per_token = torch.nn.functional.cross_entropy(
                    ce_logits_flat,
                    ce_labels_flat,
                    ignore_index=-100,
                    reduction="none",
                )
                attn_mask_flat = attn_valid.reshape(-1).float()
                ce_loss = (ce_per_token * attn_mask_flat).sum() / attn_mask_flat.sum().clamp(min=1.0)
            else:
                ce_loss = self._compute_kd_loss(model, inputs, return_outputs=return_outputs, num_items_in_batch=num_items_in_batch, student_logits=student_outputs.logits)

            loss = kd_weight * kd_loss + ce_weight * ce_loss

            with torch.no_grad():
                student_probs = F.softmax(shifted_student_logits.detach(), dim=-1)
                student_entropy = -(student_probs * torch.log(student_probs + 1e-10)).sum(dim=-1)
                
                teacher_probs = F.softmax(shifted_teacher_logits.detach(), dim=-1)
                teacher_entropy = -(teacher_probs * torch.log(teacher_probs + 1e-10)).sum(dim=-1)
                
                valid_mask = (shifted_labels != -100).float()
                mask_sum = valid_mask.sum().clamp(min=1)
                
                avg_student_entropy = (student_entropy * valid_mask).sum() / mask_sum
                avg_teacher_entropy = (teacher_entropy * valid_mask).sum() / mask_sum


                gathered_student_entropy = self.accelerator.gather(avg_student_entropy).mean()
                gathered_teacher_entropy = self.accelerator.gather(avg_teacher_entropy).mean()

            mode = "train" if model.training else "eval"
            self._metrics[mode]["student_entropy"].append(gathered_student_entropy.item())
            self._metrics[mode]["teacher_entropy"].append(gathered_teacher_entropy.item())
            self._metrics[mode]["ce_loss"].append(ce_loss.item())
            self._metrics[mode]["kd_loss"].append(kd_loss.item())

                

        empty_cache()
        return (loss, student_outputs) if return_outputs else loss


    @staticmethod
    def _phi(x: torch.Tensor, kind: str) -> torch.Tensor:
        """
        Concave regularizer φ used in the occupancy objective.
        Available kinds:
          - "identity": φ(x) = x
          - "logsigmoid": φ(x) = log(sigmoid(x)) = -softplus(-x)
          - "tanh": φ(x) = tanh(x)
          - "softplus_neg": φ(x) = -softplus(-x) (alias of logsigmoid)
          - "chi2": φ(x) = x - x^2/4
        """
        if kind == "identity":
            return x
        elif kind == "logsigmoid":
            return F.logsigmoid(x)
        elif kind == "tanh":
            return torch.tanh(x)
        elif kind == "softplus_neg":
            return -F.softplus(-x)
        elif kind == "chi2":
            return x - 0.25 * (x * x)
        else:
            raise ValueError(f"Unsupported phi kind: {kind}")

    def _apply_phi(self, x: torch.Tensor, alpha: float, phi_type: str) -> torch.Tensor:
        return self._phi(x, phi_type)


    def _compute_kd_loss(
        self, 
        model, 
        inputs, 
        return_outputs: bool = False,
        num_items_in_batch: Optional[int] = None,
        student_logits: Optional[torch.Tensor] = None,
    ):
        """
        
        Args:
        """
        device = self.accelerator.device
        
        gamma = float(getattr(self.args, "gamma", 0.9995))
        alpha = float(getattr(self.args, "alpha", 0.05))
        phi_type = getattr(self.args, "phi_type", "chi2")
        gamma_tensor = torch.tensor(gamma, dtype=torch.float32, device=device)
        

        input_ids = inputs["input_ids"]  # [batch_size, seq_len]
        attention_mask = inputs.get("attention_mask")  # [batch_size, seq_len]
     

        if attention_mask is None:
            pad_token_id = self.processing_class.pad_token_id if self.processing_class.pad_token_id is not None else 0
            attention_mask = (input_ids != pad_token_id).to(torch.long)
        
        completion_mask = (inputs["labels"] != -100) & (attention_mask.bool())   # [batch_size, seq_len], 0=prompt, 1=completion
   
        


        teacher_token_local = completion_mask.sum().to(torch.float32)
        teacher_token_sum = self.accelerator.reduce(teacher_token_local, reduction="sum")
        normalizer = teacher_token_sum / self.accelerator.num_processes
        

        if student_logits is None:
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            )
            logits = outputs.logits
        else:

            outputs = None
            logits = student_logits
        

        logits_shift = logits[:, :-1, :]
        

        labels_shift = input_ids[:, 1:]
        completion_mask_shift = completion_mask[:, 1:]
        

        valid_mask = completion_mask_shift.bool()
        

        V_si_data = torch.logsumexp(logits_shift, dim=-1)  # [batch_size, seq_len-1]
        

        l_ai_si = torch.gather(
            logits_shift,
            dim=-1,
            index=labels_shift.clamp(min=0).unsqueeze(-1)
        ).squeeze(-1)  # [batch_size, seq_len-1]
        

        l_ai_si = torch.where(valid_mask, l_ai_si, torch.zeros_like(l_ai_si))
        V_si_data = torch.where(valid_mask, V_si_data, torch.zeros_like(V_si_data))
        

        V_next_data = torch.zeros_like(V_si_data)
        V_next_data[:, :-1] = V_si_data[:, 1:].clone()
        V_next_data = torch.where(valid_mask, V_next_data, torch.zeros_like(V_next_data))
        

        step_idx = torch.cumsum(valid_mask.to(torch.int32), dim=1) - 1
        step_idx = torch.where(valid_mask, step_idx, torch.full_like(step_idx, -1))
        gamma_pow = torch.pow(gamma_tensor, step_idx.clamp(min=0))
        gamma_pow = torch.where(valid_mask, gamma_pow, torch.zeros_like(gamma_pow))
        

        term1_arg = alpha * (l_ai_si - gamma * V_next_data)
        term1 = (1.0 / alpha) * self._apply_phi(term1_arg, alpha, phi_type)
        term2_data = -(V_si_data - gamma * V_next_data)
        
        local_data_loss_sum = (gamma_pow * (term1 + term2_data) * valid_mask).sum()
        

        last_step = step_idx.max(dim=1, keepdim=True).values
        last_mask = valid_mask & (step_idx == last_step)
        V_last_data = torch.where(last_mask, V_si_data, torch.zeros_like(V_si_data)).sum(dim=1)
        gamma_last_data = torch.where(last_mask, gamma_pow, torch.zeros_like(gamma_pow)).sum(dim=1)
        
        eps = 1e-8
        one_minus_gamma = max(1.0 - gamma, eps)


        has_valid = valid_mask.any(dim=1)  # [batch_size]


        terminal_data = torch.zeros(input_ids.size(0), device=device)
        if has_valid.any():
            valid_indices = has_valid.nonzero(as_tuple=True)[0]
            terminal_data[valid_indices] = (
                (gamma_last_data[valid_indices] / (alpha * one_minus_gamma))
                * self._apply_phi(alpha * one_minus_gamma * V_last_data[valid_indices], alpha, phi_type)
                - gamma_last_data[valid_indices] * V_last_data[valid_indices]
            )

        terminal_obj = terminal_data[has_valid].mean()
    

        del logits, logits_shift, V_si_data, V_next_data
        del gamma_pow, step_idx, valid_mask
        torch.cuda.empty_cache()
        

        data_obj = local_data_loss_sum / normalizer
        loss = -(data_obj + terminal_obj)
        
        if return_outputs:
            return (loss, outputs)
        else:
            return loss



    @staticmethod
    def generate_on_policy_outputs(model, inputs, generation_config, pad_token_id=None):
        synced_gpus = dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1
        # Generate output with respect to the prompt-only
        generated_outputs = model.generate(
            input_ids=inputs["prompts"],
            attention_mask=inputs.get("prompt_attention_mask", None),
            generation_config=generation_config,
            return_dict_in_generate=True,
            synced_gpus=synced_gpus,
            use_cache=True,
        )

        # Get the generated token IDs
        generated_tokens = generated_outputs.sequences
        # Calculate new attention mask
        new_attention_mask = torch.ones_like(generated_tokens)
        new_labels = generated_tokens.clone()

        # If there's pad_token_id, set attention mask to 0 for padding tokens
        if pad_token_id is not None:
            new_labels[new_labels == pad_token_id] = -100
            new_attention_mask[generated_tokens == pad_token_id] = 0

        return generated_tokens, new_attention_mask, new_labels

    def _should_generate_on_policy(self) -> bool:
        if self.lmbda <= 0:
            return False
        if self.lmbda >= 1:
            return True

        sample = torch.rand(1, device=self.accelerator.device)
        if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
            dist.broadcast(sample, src=0)
        return bool(sample.item() < self.lmbda)

    def training_step(
        self, model: nn.Module, inputs: dict[str, Union[torch.Tensor, Any]], num_items_in_batch: Optional[int] = None
    ) -> torch.Tensor:
        """
        Perform a training step for the Generalized Knowledge Distillation (GKD) model.

        This method implements the on-policy learning approach described in the GKD paper. With probability
        `self.lmbda`, it generates new responses using the student model, which are then used for training instead of
        the original inputs.
        """
        if self.seq_kd:
            with unwrap_model_for_generation(self.teacher_model, self.accelerator) as unwrapped_model:
                new_input_ids, new_attention_mask, new_labels = self.generate_on_policy_outputs(
                    unwrapped_model, inputs, self.generation_config, self.processing_class.pad_token_id
                )
            inputs["input_ids"] = new_input_ids
            inputs["attention_mask"] = new_attention_mask
            inputs["labels"] = new_labels
        if self._should_generate_on_policy():
            with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
                new_input_ids, new_attention_mask, new_labels = self.generate_on_policy_outputs(
                    unwrapped_model, inputs, self.generation_config, self.processing_class.pad_token_id
                )
            inputs["input_ids"] = new_input_ids
            inputs["attention_mask"] = new_attention_mask
            inputs["labels"] = new_labels

        loss = super().training_step(model, inputs, num_items_in_batch)
        return loss
