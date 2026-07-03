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

import warnings

import torch
import torch.nn.functional as F

from .gkd_trainer import GKDTrainer
from .utils import empty_cache


class HybridGKDTrainer(GKDTrainer):
    """On-policy GKD variant that mixes teacher-distribution KD with teacher-argmax hard labels."""

    _tag_names = ["trl", "gkd", "hybrid_gkd"]
    _name = "HybridGKD"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if getattr(self.args, "lambda_sft", 0.0) <= 0:
            warnings.warn(
                "HybridGKDTrainer was initialized with `lambda_sft<=0`, so the hard-label term is disabled. "
                "Set `lambda_sft>0` to enable teacher-argmax hybrid supervision."
            )

    @staticmethod
    def teacher_argmax_loss(student_logits, teacher_logits, labels):
        if student_logits.size(-1) != teacher_logits.size(-1):
            common_vocab_size = min(student_logits.size(-1), teacher_logits.size(-1))
            student_logits = student_logits[..., :common_vocab_size]
            teacher_logits = teacher_logits[..., :common_vocab_size]

        hard_labels = teacher_logits.argmax(dim=-1)
        valid_mask = labels != -100
        if not valid_mask.any():
            return student_logits.sum() * 0.0

        return F.cross_entropy(student_logits[valid_mask], hard_labels[valid_mask], reduction="mean")

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if self.use_liger_gkd_loss:
            raise NotImplementedError("HybridGKDTrainer does not support `use_liger_kernel=True`.")

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

        hard_loss = self.teacher_argmax_loss(shifted_student_logits, shifted_teacher_logits, shifted_labels)

        kd_weight = getattr(self.args, "lambda_fkl", 1.0)
        hard_weight = getattr(self.args, "lambda_sft", 1.0)
        loss = kd_weight * kd_loss + hard_weight * hard_loss

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
        self._metrics[mode]["hard_loss"].append(hard_loss.item())
        self._metrics[mode]["kd_loss"].append(kd_loss.item())

        empty_cache()
        return (loss, student_outputs) if return_outputs else loss
