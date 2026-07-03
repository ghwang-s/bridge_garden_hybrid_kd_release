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

import contextlib
import math
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, Union
import torch.nn.functional as F

import torch
import torch.nn as nn
from accelerate import PartialState, logging
from datasets import Dataset, IterableDataset
from transformers import (
    AutoProcessor,
    BaseImageProcessor,
    DataCollator,
    FeatureExtractionMixin,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    ProcessorMixin,
    TrainingArguments,
    AutoModelForCausalLM
)
from transformers.data.data_collator import DataCollatorMixin
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import EvalPrediction
from transformers.utils import is_peft_available
from ..models import prepare_deepspeed

from ..data_utils import (
    apply_chat_template,
    is_conversational,
    is_conversational_from_value,
    maybe_convert_to_chatml,
    pack_dataset,
    prepare_multimodal_messages,
    truncate_dataset,
)
from ..models import clone_chat_template, get_act_offloading_ctx_manager, prepare_peft_model
from .base_trainer import BaseTrainer
from .sft_config import SFTConfig
from .utils import (
    create_model_from_path,
    entropy_from_logits,
    flush_left,
    get_config_model_id,
    pad,
    remove_none_values,
    selective_log_softmax,
)


if is_peft_available():
    from peft import PeftConfig, PeftModel, PeftType


logger = logging.get_logger(__name__)


FLASH_ATTENTION_VARIANTS = {
    "flash_attention_2",
    "flash_attention_3",
    "kernels-community/flash-attn",
    "kernels-community/vllm-flash-attn3",
    "kernels-community/flash-attn3",
}

TEMPERATURE_SCHEDULE_PHI_TYPES = {
    "temperature_schedule_high_to_low",
    "temperature_schedule_low_to_high",
}


def get_dataset_column_names(dataset: Union[Dataset, IterableDataset]) -> list[str]:
    return list(next(iter(dataset)).keys()) if dataset.column_names is None else dataset.column_names


@dataclass
class DataCollatorForLanguageModeling(DataCollatorMixin):
    """
    Data collator used for language modeling data. Inputs are dynamically padded to the maximum length of a batch.

    This collator expects each example in the input list to be a dictionary containing at least the `"input_ids"` key.
    If the input contains a `"completion_mask"`, it is used to set the labels to `-100` for tokens that are not in the
    completion. If `"assistant_masks"` are present, they are used to set the labels to `-100` for tokens that are not
    in the assistant part of the sequence. The collator returns a dictionary containing the following keys:
    - `"input_ids"`: Tensor of input IDs, padded to the maximum length of the batch.
    - `"labels"`: Tensor of labels, padded to the maximum length of the batch. If `completion_only_loss` is set to
    `True`, tokens that are not in the completion are set to -100. If `assistant_masks` are present, tokens that are
    not in the assistant part of the sequence are set to -100. If `padding_free` is set to `False`, the following key
    is also returned:
    - `"attention_mask"`: Tensor of attention masks, padded to the maximum length of the batch.
    If `padding_free` is set to `True`, the following key is also returned:
    - `"position_ids"`: Tensor of position IDs, padded to the maximum length of the batch.

    Args:
        pad_token_id (`int`):
            Token ID to use for padding.
        completion_only_loss (`bool`, *optional*, defaults to `True`):
            When the input contains a completion mask (`completion_mask`), the labels are set to -100 for the tokens
            that are no in the completion.
        padding_free (`bool`, *optional*, defaults to `False`):
            If set to `True`, the sequences will be flattened into a single sequence, and the position IDs will be
            generated accordingly and returned instead of the attention mask.
        pad_to_multiple_of (`int`, *optional*):
            If set, the sequences will be padded to a multiple of this value.
        return_tensors (`str`, *optional*, defaults to `"pt"`):
            Type of Tensor to return. Only `"pt"` is currently supported.

    Examples:
    ```python
    >>> from trl.trainer.sft_trainer import DataCollatorForLanguageModeling

    >>> collator = DataCollatorForLanguageModeling(pad_token_id=0)
    >>> examples = [{"input_ids": [1, 2, 3]}, {"input_ids": [4, 5]}]
    >>> collator(examples)
    {'input_ids': tensor([[  1,  2,  3],
                          [  4,  5,  0]]),
     'attention_mask': tensor([[  1,  1,  1],
                               [  1,  1,  0]]),
     'labels': tensor([[   1,    2,    3],
                       [   4,    5, -100]])}

    >>> # With completion mask
    >>> examples = [
    ...     {"input_ids": [1, 2, 3], "completion_mask": [0, 1, 1]},
    ...     {"input_ids": [4, 5], "completion_mask": [0, 1]},
    ... ]
    >>> collator(examples)
    {'input_ids': tensor([[  1,  2,  3],
                          [  4,  5,  0]]),
     'attention_mask': tensor([[  1,  1,  1],
                               [  1,  1,  0]]),
     'labels': tensor([[-100,    2,    3],
                       [-100,    5, -100]])}

    >>> # With padding_free
    >>> collator = DataCollatorForLanguageModeling(pad_token_id=0, padding_free=True)
    >>> collator(examples)
    {'input_ids': tensor([[ 1, 2, 3, 4, 5]]),
     'position_ids': tensor([[0, 1, 2, 0, 1]]),
     'labels': tensor([[1, 2, 3, 4, 5]])}
    ```
    """

    pad_token_id: int
    completion_only_loss: bool = True
    padding_free: bool = False
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def torch_call(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        # Convert to tensor
        input_ids = [torch.tensor(example["input_ids"]) for example in examples]
        if "labels" in examples[0]:
            labels = [torch.tensor(example["labels"]) for example in examples]
        else:
            labels = [torch.tensor(example["input_ids"]) for example in examples]

        # For padding-free, we should NOT create attention_mask as it causes FlashAttention to ignore position_ids and
        # compute wrong cu_seq_lens from the all-1s mask
        if self.padding_free:
            if "seq_lengths" in examples[0]:
                position_ids = self.get_position_ids_from_packed_seq_lengths(
                    [example["seq_lengths"] for example in examples]
                )
            else:
                position_ids = [torch.arange(len(ids)) for ids in input_ids]
        else:
            attention_mask = [torch.ones_like(ids) for ids in input_ids]
        if self.completion_only_loss and "completion_mask" in examples[0]:
            completion_mask = [torch.tensor(example["completion_mask"]) for example in examples]
        if "assistant_masks" in examples[0]:
            assistant_masks = [torch.tensor(example["assistant_masks"]) for example in examples]

        # If padding_free, flatten everything into a single sequence
        output = {}
        if self.padding_free:
            input_ids = [torch.cat(input_ids, dim=0)]
            labels = [torch.cat(labels, dim=0)]
            position_ids = [torch.cat(position_ids, dim=0)]
            if self.completion_only_loss and "completion_mask" in examples[0]:
                completion_mask = [torch.cat(completion_mask, dim=0)]
            if "assistant_masks" in examples[0]:
                assistant_masks = [torch.cat(assistant_masks, dim=0)]

        # Pad
        output["input_ids"] = pad(
            input_ids,
            padding_value=self.pad_token_id,
            padding_side="right",
            pad_to_multiple_of=self.pad_to_multiple_of,
        )
        output["labels"] = pad(
            labels, padding_value=-100, padding_side="right", pad_to_multiple_of=self.pad_to_multiple_of
        )
        if self.padding_free:
            output["position_ids"] = pad(
                position_ids, padding_value=0, padding_side="right", pad_to_multiple_of=self.pad_to_multiple_of
            )
            output["labels"][output["position_ids"] == 0] = -100
        else:
            output["attention_mask"] = pad(
                attention_mask, padding_value=0, padding_side="right", pad_to_multiple_of=self.pad_to_multiple_of
            )
        if self.completion_only_loss and "completion_mask" in examples[0]:
            completion_mask = pad(
                completion_mask, padding_value=0, padding_side="right", pad_to_multiple_of=self.pad_to_multiple_of
            )
            output["labels"][completion_mask == 0] = -100  # mask everything that is not in the completion
        if "assistant_masks" in examples[0]:
            assistant_masks = pad(
                assistant_masks, padding_value=0, padding_side="right", pad_to_multiple_of=self.pad_to_multiple_of
            )
            output["labels"][assistant_masks == 0] = -100
        return output

    @staticmethod
    def get_position_ids_from_packed_seq_lengths(batch_seq_lengths: list[list[int]]) -> list[torch.Tensor]:
        """
        Get position IDs for packed sequences.

        Args:
            batch_seq_lengths (`list[list[int]]`):
                A list of lists containing the lengths of each individual document in the packed batch.

        Return:
            `list[torch.Tensor]`:
                A list of tensors containing the position IDs for each packed sequence.
        """
        # Get lengths per row
        example_lengths = [sum(seq_lengths) for seq_lengths in batch_seq_lengths]
        # Flat list of lengths
        batch_seq_lengths = torch.tensor(
            [seq_length for seq_lengths in batch_seq_lengths for seq_length in seq_lengths]
        )
        position_ids = torch.ones(sum(example_lengths), dtype=batch_seq_lengths.dtype)
        position_ids[0] = 0
        # Reset position ids to 0 at the start of each sequence
        position_ids[batch_seq_lengths[:-1].cumsum(0)] = -(batch_seq_lengths[:-1] - 1)
        position_ids = position_ids.cumsum(0)
        # Split back into one tensor per example
        return list(position_ids.split(example_lengths))


@dataclass
class DataCollatorForVisionLanguageModeling(DataCollatorMixin):
    """
    Data collator for vision-language modeling tasks.

    Unlike text-only datasets—where the collator typically receives pre-tokenized inputs ready for batching,
    vision-language data processing involves converting images into pixel values. This conversion is disk-intensive,
    making upfront preprocessing of the entire dataset impractical. Therefore, this collator performs tokenization and
    image processing on-the-fly to efficiently prepare batches.

    Each input example should be a dictionary containing at least:
    - An `"images"` key holding the image data.
    - [language modeling](#language-modeling) type: either a `"messages"` key for conversational inputs or a `"text"`
      key for standard text inputs.
    - [prompt-completion](#prompt-completion) type: keys `"prompt"` and `"completion"` for the prompt and completion.

    The collator outputs a dictionary including:
    - `"input_ids"`: Tensor of token IDs.
    - `"attention_mask"`: Tensor indicating attention mask.
    - `"pixel_values"`: Tensor representing image pixel values.
    - `"labels"`: Tensor for training labels.

    Additional keys may be present depending on the processor, such as `"image_grid_thw"`.

    Args:
        processor ([`~transformers.ProcessorMixin`]):
            The processor used to tokenize text and process images. It must be a subclass of
            [`~transformers.ProcessorMixin`] and include a `tokenizer` with a defined `pad_token_id`.
        max_length (`int` or `None`, optional, defaults to `None`):
            Maximum sequence length for input tokens. If `None`, no truncation is applied.
        completion_only_loss (`bool`, *optional*, defaults to `False`):
            Whether to compute loss only on the completion part of the sequence. When `True`, the labels for the prompt
            part are set to -100. It requires the dataset type to be prompt-completion.
        pad_to_multiple_of (`int` or `None`, optional, defaults to `None`):
            If set, the sequences will be padded to a multiple of this value.
        dataset_text_field (`str`, optional, defaults to `"text"`):
            Name of the column that contains text data in the dataset. This parameter is only relevant for [standard
            datasets format](dataset_formats#standard).
        return_tensors (`str`, optional, defaults to `"pt"`):
            The tensor type to return. Currently, only `"pt"` (PyTorch tensors) is supported.

    Example:
    ```python
    >>> from trl.trainer.sft_trainer import DataCollatorForVisionLanguageModeling
    >>> from transformers import AutoProcessor

    >>> processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
    >>> collator = DataCollatorForVisionLanguageModeling(processor)
    >>> examples = [
    ...     {"images": [Image.open("image_0.png")], "messages": [{"role": "user", "content": "What is this?"}]},
    ...     {"images": [Image.open("image_1.png")], "messages": [{"role": "user", "content": "Describe this image."}]},
    ... ]
    >>> collator(examples)
    {'input_ids': tensor([[151644,   8948,    198,   2610,    525,    264,  10950,  17847,     13,  151645,    198,
                           151644,    872,    198, 151652, 151655, 151655, 151655,  151655, 151653,   3838,    374,
                              419,     30, 151645,    198],
                          [151644,   8948,    198,   2610,    525,    264,  10950,  17847,     13,  151645,    198,
                           151644,    872,    198, 151652, 151655, 151655, 151655,  151655, 151653,  74785,    419,
                             2168,     13, 151645,    198]]),
     'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                               [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]),
     'pixel_values': tensor([[-0.9893,  0.1785,  1.5362,  ..., -0.0582,  0.8661, -0.2431],
                             [-0.2302,  0.9522, -1.1061,  ...,  0.0555,  1.3354, -0.6412],
                             [ 1.2150,  0.9084,  0.7041,  ...,  0.2404, -0.8403, -0.5133],
                             ...,
                             [ 0.6895,  0.2807,  0.2515,  ..., -0.2004, -1.2100,  0.0555],
                             [ 0.8209, -0.9748,  1.5654,  ...,  1.6055, -0.4706,  0.5817],
                             [-1.0915,  0.4559,  0.9230,  ...,  0.5106,  0.0982, -0.1720]]),
     'image_grid_thw': tensor([[1, 4, 4],
                               [1, 4, 4]]),
     'labels': tensor([[151644,   8948,    198,   2610,    525,    264,  10950,  17847,     13,  151645,    198,
                        151644,    872,    198, 151652, 151655, 151655, 151655,  151655, 151653,   3838,    374,
                           419,     30, 151645,    198],
                        [151644,   8948,    198,   2610,    525,    264,  10950,  17847,     13,  151645,    198,
                         151644,    872,    198, 151652, 151655, 151655, 151655,  151655, 151653,  74785,    419,
                           2168,     13, 151645,    198]])}
    ```
    """

    processor: ProcessorMixin
    max_length: Optional[int] = None
    completion_only_loss: bool = False  # default not used in practice; SFTTrainer always passes the relevant value
    pad_to_multiple_of: Optional[int] = None
    dataset_text_field: str = "text"
    return_tensors: str = "pt"

    def torch_call(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        if "messages" in examples[0] or self.dataset_text_field in examples[0]:
            if self.completion_only_loss:
                raise ValueError(
                    "The `completion_only_loss` argument is not supported for language modeling datasets."
                )
            return self._collate_language_modeling(examples)
        elif "prompt" in examples[0] and "completion" in examples[0]:
            return self._collate_prompt_completion(examples)
        else:
            raise KeyError(f"Unexpected input keys in examples: {list(examples[0].keys())}.")

    def _collate_language_modeling(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        images = [example["images"] for example in examples]
        # Transformers requires at least one image in the batch, otherwise it throws an error
        if all(img_list == [] for img_list in images):
            images = None

        if "messages" in examples[0]:  # conversational case
            messages = [prepare_multimodal_messages(example["messages"], example["images"]) for example in examples]
            texts = self.processor.apply_chat_template(messages)
        elif self.dataset_text_field in examples[0]:  # standard case
            texts = [example[self.dataset_text_field] for example in examples]
        else:
            raise KeyError(
                "The input examples must contain either 'messages' for conversational data or 'text' for standard "
                "data."
            )

        output = self.processor(
            images=images,
            text=texts,
            padding=True,
            padding_side="right",
            pad_to_multiple_of=self.pad_to_multiple_of,
            truncation=self.max_length is not None,
            max_length=self.max_length,
            return_tensors=self.return_tensors,
            add_special_tokens=False,  # to avoid adding the BOS, twice see https://huggingface.co/blog/qgallouedec/gotchas-in-tokenizer-behavior#7-chat-template-and-tokenization-dont-compose-due-to-special-tokens
        )
        labels = output["input_ids"].clone()
        labels[output["attention_mask"] == 0] = -100
        # We mask only padding tokens (-100) in the labels. Vision tokens are left unchanged because their handling in
        # loss computation has to be done by the model, and masking them here would be infeasible in practice as vision
        # token definitions vary across architectures.
        output["labels"] = labels
        return output

    def _collate_prompt_completion(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        if self.pad_to_multiple_of is not None:
            raise NotImplementedError(
                "Padding to a multiple of a value is not yet implemented for vision-language modeling and "
                "prompt-completion data yet."
            )
        images = [example["images"] for example in examples]
        # Transformers requires at least one image in the batch, otherwise it throws an error
        if all(img_list == [] for img_list in images):
            images = None
        if is_conversational(examples[0]):  # conversational case
            for example in examples:
                example["prompt"] = prepare_multimodal_messages(example["prompt"], images=example["images"])
                example["completion"] = prepare_multimodal_messages(example["completion"], images=[])
            examples = [apply_chat_template(example, self.processor) for example in examples]

        prompts = [example["prompt"] for example in examples]
        completions = [example["completion"] for example in examples]

        processed_prompts = self.processor(
            images=images,
            text=prompts,
            padding=True,
            padding_side="left",
            return_tensors=self.return_tensors,
            add_special_tokens=False,  # to avoid adding the BOS, twice see https://huggingface.co/blog/qgallouedec/gotchas-in-tokenizer-behavior#7-chat-template-and-tokenization-dont-compose-due-to-special-tokens
        )
        processed_completions = self.processor(
            text=completions,
            padding=True,
            padding_side="right",
            return_tensors=self.return_tensors,
            add_special_tokens=False,  # to avoid adding the BOS, twice see https://huggingface.co/blog/qgallouedec/gotchas-in-tokenizer-behavior#7-chat-template-and-tokenization-dont-compose-due-to-special-tokens
        )

        # Concatenate prompts and completions
        prompt_ids, completion_ids = processed_prompts["input_ids"], processed_completions["input_ids"]
        prompt_mask, completion_mask = processed_prompts["attention_mask"], processed_completions["attention_mask"]
        input_ids = torch.cat((prompt_ids, completion_ids), dim=1)
        attention_mask = torch.cat((prompt_mask, completion_mask), dim=1)
        completion_mask = torch.cat((torch.zeros_like(prompt_mask), completion_mask), dim=1)
        if "token_type_ids" in processed_prompts:  # special case for Gemma
            prompt_token_type_ids = processed_prompts["token_type_ids"]
            completion_token_type_ids = processed_completions["token_type_ids"]
            token_type_ids = torch.cat((prompt_token_type_ids, completion_token_type_ids), dim=1)

        # Flush left to reduce padding
        if "token_type_ids" in processed_prompts:
            attention_mask, input_ids, completion_mask, token_type_ids = flush_left(
                attention_mask, input_ids, completion_mask, token_type_ids
            )
        else:
            attention_mask, input_ids, completion_mask = flush_left(attention_mask, input_ids, completion_mask)

        # Truncate if necessary
        if self.max_length is not None:
            input_ids = input_ids[:, : self.max_length]
            attention_mask = attention_mask[:, : self.max_length]
            completion_mask = completion_mask[:, : self.max_length]
            if "token_type_ids" in processed_prompts:
                token_type_ids = token_type_ids[:, : self.max_length]

        # Create labels and mask padding tokens
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        if self.completion_only_loss:
            labels[completion_mask == 0] = -100

        # Build the output dictionary
        output = processed_prompts  # we take processed_prompts because it contains the images
        output["input_ids"] = input_ids
        output["attention_mask"] = attention_mask
        output["labels"] = labels
        if "token_type_ids" in processed_prompts:
            output["token_type_ids"] = token_type_ids
        return output


def dft_loss(outputs, labels, num_items_in_batch=None):
    """
    DFT loss function, as presented in [On the Generalization of SFT: A Reinforcement Learning Perspective with Reward
    Rectification](https://huggingface.co/papers/2508.05629)
    """
    labels = nn.functional.pad(labels, (0, 1), value=-100)
    shift_labels = labels[..., 1:].contiguous()
    loss_mask = shift_labels != -100
    shift_labels[~loss_mask] = 0
    logprobs = selective_log_softmax(outputs.logits, shift_labels)
    per_token_loss = -logprobs.exp().detach() * logprobs
    if num_items_in_batch is None:
        num_items_in_batch = loss_mask.sum()
    loss = (per_token_loss * loss_mask).sum() / num_items_in_batch
    return loss


class SFTTrainer(BaseTrainer):
    """
    Trainer for Supervised Fine-Tuning (SFT) method.

    This class is a wrapper around the [`~transformers.Trainer`] class and inherits all of its attributes and methods.

    Example:

    ```python
    from datasets import load_dataset
    from trl import SFTTrainer

    dataset = load_dataset("roneneldan/TinyStories", split="train[:1%]")

    trainer = SFTTrainer(model="Qwen/Qwen2-0.5B-Instruct", train_dataset=dataset)
    trainer.train()
    ```

    Args:
        model (`Union[str, PreTrainedModel]`):
            Model to be trained. Can be either:

            - A string, being the *model id* of a pretrained model hosted inside a model repo on huggingface.co, or a
              path to a *directory* containing model weights saved using
              [`~transformers.PreTrainedModel.save_pretrained`], e.g., `'./my_model_directory/'`. The model is loaded
              using `<ModelArchitecture>.from_pretrained` (where `<ModelArchitecture>` is derived from the model
              config) with the keyword arguments in `args.model_init_kwargs`.
            - A [`~transformers.PreTrainedModel`] object.
            If you're training a model with an MoE architecture and want to include the load balancing/auxilliary loss
            as a part of the final loss, remember to set the `output_router_logits` config of the model to `True`.
        args ([`SFTConfig`], *optional*):
            Configuration for this trainer. If `None`, a default configuration is used.
        data_collator ([`~transformers.DataCollator`], *optional*):
            Function to use to form a batch from a list of elements of the processed `train_dataset` or `eval_dataset`.
            Will default to [`~trainer.sft_trainer.DataCollatorForLanguageModeling`] if the model is a language model
            and [`~trainer.sft_trainer.DataCollatorForVisionLanguageModeling`] if the model is a vision-language model.
        train_dataset ([`~datasets.Dataset`] or [`~datasets.IterableDataset`]):
            Dataset to use for training. SFT supports both [language modeling](#language-modeling) type and
            [prompt-completion](#prompt-completion) type. The format of the samples can be either:

            - [Standard](dataset_formats#standard): Each sample contains plain text.
            - [Conversational](dataset_formats#conversational): Each sample contains structured messages (e.g., role
              and content).

            The trainer also supports processed datasets (tokenized) as long as they contain an `input_ids` field.
        eval_dataset ([`~datasets.Dataset`], [`~datasets.IterableDataset`] or `dict[str, Union[Dataset, IterableDataset]]`):
            Dataset to use for evaluation. It must meet the same requirements as `train_dataset`.
        processing_class ([`~transformers.PreTrainedTokenizerBase`], [`~transformers.ProcessorMixin`], *optional*):
            Processing class used to process the data. If `None`, the processing class is loaded from the model's name
            with [`~transformers.AutoProcessor.from_pretrained`]. A padding token, `tokenizer.pad_token`, must be set.
            If the processing class has not set a padding token, `tokenizer.eos_token` will be used as the default.
        compute_loss_func (`Callable`, *optional*):
            A function that accepts the raw model outputs, labels, and the number of items in the entire accumulated
            batch (batch_size * gradient_accumulation_steps) and returns the loss. For example, see the default [loss
            function](https://github.com/huggingface/transformers/blob/052e652d6d53c2b26ffde87e039b723949a53493/src/transformers/trainer.py#L3618)
            used by [`Trainer`].
        compute_metrics (`Callable[[EvalPrediction], dict]`, *optional*):
            The function that will be used to compute metrics at evaluation. Must take a
            [`~transformers.EvalPrediction`] and return a dictionary string to metric values. When passing
            [`SFTConfig`] with `batch_eval_metrics` set to `True`, your `compute_metrics` function must take a boolean
            `compute_result` argument. This will be triggered after the last eval batch to signal that the function
            needs to calculate and return the global summary statistics rather than accumulating the batch-level
            statistics.
        callbacks (list of [`~transformers.TrainerCallback`], *optional*):
            List of callbacks to customize the training loop. Will add those to the list of default callbacks detailed
            in [here](https://huggingface.co/docs/transformers/main_classes/callback).

            If you want to remove one of the default callbacks used, use the [`~transformers.Trainer.remove_callback`]
            method.
        optimizers (`tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]]`, *optional*, defaults to `(None, None)`):
            A tuple containing the optimizer and the scheduler to use. Will default to an instance of `AdamW` on your
            model and a scheduler given by [`~transformers.get_linear_schedule_with_warmup`] controlled by `args`.
        optimizer_cls_and_kwargs (`tuple[Type[torch.optim.Optimizer], Dict[str, Any]]`, *optional*):
            A tuple containing the optimizer class and keyword arguments to use. Overrides `optim` and `optim_args` in
            `args`. Incompatible with the `optimizers` argument.

            Unlike `optimizers`, this argument avoids the need to place model parameters on the correct devices before
            initializing the Trainer.
        preprocess_logits_for_metrics (`Callable[[torch.Tensor, torch.Tensor], torch.Tensor]`, *optional*):
            A function that preprocess the logits right before caching them at each evaluation step. Must take two
            tensors, the logits and the labels, and return the logits once processed as desired. The modifications made
            by this function will be reflected in the predictions received by `compute_metrics`.

            Note that the labels (second parameter) will be `None` if the dataset does not have them.
        peft_config ([`~peft.PeftConfig`], *optional*):
            PEFT configuration used to wrap the model. If `None`, the model is not wrapped.
        formatting_func (`Callable`, *optional*):
            Formatting function applied to the dataset before tokenization. Applying the formatting function explicitly
            converts the dataset into a [language modeling](#language-modeling) type.
    """

    _tag_names = ["trl", "sft"]
    _name = "SFT"

    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        teacher_model: Union[PreTrainedModel, nn.Module, str] = None,
        args: Optional[Union[SFTConfig, TrainingArguments]] = None,
        data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, dict[str, Dataset]]] = None,
        processing_class: Optional[Union[PreTrainedTokenizerBase, ProcessorMixin]] = None,
        compute_loss_func: Optional[Callable] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], dict]] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (None, None),
        optimizer_cls_and_kwargs: Optional[tuple[type[torch.optim.Optimizer], dict[str, Any]]] = None,
        preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        peft_config: Optional["PeftConfig"] = None,
        formatting_func: Optional[Callable[[dict], str]] = None,
    ):
        # Args
        if args is None:
            model_name = model if isinstance(model, str) else get_config_model_id(model.config)
            model_name = model_name.split("/")[-1]
            args = SFTConfig(f"{model_name}-SFT")
        elif isinstance(args, TrainingArguments) and not isinstance(args, SFTConfig):
            dict_args = args.to_dict()
            dict_args["hub_token"] = args.hub_token  # to_dict hides the hub_token
            dict_args.pop("push_to_hub_token")
            args = SFTConfig(**dict_args)

        # Model
        if isinstance(model, str):
            model = create_model_from_path(model, **args.model_init_kwargs or {})
        else:
            if args.model_init_kwargs is not None:
                logger.warning(
                    "You passed `model_init_kwargs` to the `SFTConfig`, but your model is already instantiated. "
                    "The `model_init_kwargs` will be ignored."
                )

        # Processing class
        if processing_class is None:
            processing_class = AutoProcessor.from_pretrained(get_config_model_id(model.config))

        # Handle pad token for processors or tokenizers
        if isinstance(processing_class, ProcessorMixin):
            tokenizer = processing_class.tokenizer
            self._is_vlm = True
        elif isinstance(processing_class, PreTrainedTokenizerBase):
            tokenizer = processing_class
            self._is_vlm = False
        else:
            raise TypeError("The `processing_class` must be either a `PreTrainedTokenizerBase` or a `ProcessorMixin`")

        if args.eos_token is not None:
            eos_token = args.eos_token
            eos_token_id = tokenizer.convert_tokens_to_ids(eos_token)
            if eos_token_id is None:
                raise ValueError(
                    f"The specified `eos_token` ('{eos_token}') is not found in the vocabulary of the given "
                    f"`processing_class` ({processing_class.__class__.__name__}). Ensure that the `eos_token` exists "
                    "in the vocabulary before using it as an EOS token."
                )
            tokenizer.eos_token_id = eos_token_id

        if args.chat_template_path is not None:
            if os.path.isfile(args.chat_template_path) and args.chat_template_path.endswith((".jinja", ".j2")):
                with open(args.chat_template_path, encoding="utf-8") as chat_template_file:
                    processing_class.chat_template = chat_template_file.read()
                added_tokens = []
            else:
                model, processing_class, added_tokens = clone_chat_template(
                    model, processing_class, args.chat_template_path
                )
        else:
            added_tokens = []

        # Catch some wrong configurations related to VLMs
        if self._is_vlm and args.packing:
            raise ValueError(
                "Packing is not supported for vision-language models. Please set `packing=False` in the SFTConfig."
            )
        if self._is_vlm and args.padding_free:
            raise ValueError(
                "Padding-free training is yet not supported for vision-language models. Please set "
                "`padding_free=False` in the `SFTConfig`."
            )
        if self._is_vlm and args.assistant_only_loss:
            raise ValueError(
                "Assistant-only loss is not yet supported for vision-language models. Please set "
                "`assistant_only_loss=False` in the `SFTConfig`."
            )

        # PEFT configuration and model wrapping
        if peft_config is not None:
            if added_tokens:
                # Ensure that the added tokens are trainable
                if peft_config.trainable_token_indices is None:
                    peft_config.trainable_token_indices = {"embed_tokens": added_tokens}
                elif "embed_tokens" not in peft_config.trainable_token_indices:
                    peft_config.trainable_token_indices["embed_tokens"] = added_tokens
                else:
                    peft_config.trainable_token_indices["embed_tokens"].extend(added_tokens)

                # Ensure that the lm_head is trainable
                if peft_config.modules_to_save is None or "lm_head" not in peft_config.modules_to_save:
                    logger.warning(
                        "Cloning chat template added new tokens to the tokenizer, but 'lm_head' is not in PEFT's "
                        "`modules_to_save`. As a result, the model may not learn to generate outputs with these new "
                        "tokens, leading to degraded generation quality. To fix this, add "
                        "`modules_to_save=['lm_head']` to your PEFT configuration."
                    )

                    if peft_config.modules_to_save is None:
                        peft_config.modules_to_save = ["lm_head"]
                    else:
                        peft_config.modules_to_save.append("lm_head")

        # In Prompt Tuning a small set of trainable virtual tokens (continuous prompt embeddings) is prepended to the
        # input. We store the number of these tokens so we can account for them correctly when calculating accuracy.
        self.num_virtual_tokens = 0

        if peft_config is not None or (is_peft_available() and isinstance(model, PeftModel)):
            model = prepare_peft_model(model, peft_config, args)
            if model.active_adapter in model.peft_config:
                peft_model_config = model.peft_config[model.active_adapter]
                self.num_virtual_tokens = getattr(peft_model_config, "num_virtual_tokens", 0)

        # Data collator
        # BFD packing requires padding-free mode; otherwise, the collator outputs padded attention masks, causing
        # FlashAttention to ignore position_ids and recompute them incorrectly from the padded attention mask.
        self.padding_free = args.padding_free or (args.packing and args.packing_strategy == "bfd")
        use_flash_attention = model.config._attn_implementation in FLASH_ATTENTION_VARIANTS
        if self.padding_free:
            if data_collator is not None:
                raise ValueError("Passing a custom data collator is not supported when using padding-free.")
            if args.packing and args.packing_strategy == "wrapped":
                logger.warning(
                    "You are passing `padding_free=True` with the 'wrapped' packing strategy, which is not "
                    "recommended. Please refer to the documentation to understand why this is not recommended."
                )
            if not use_flash_attention:
                logger.warning(
                    "Padding-free training is enabled, but the attention implementation is not set to a supported "
                    "flash attention variant. Padding-free training flattens batches into a single sequence, and only "
                    "the following implementations are known to reliably support this: "
                    f"{', '.join(sorted(FLASH_ATTENTION_VARIANTS))}. Using other implementations may lead to "
                    "unexpected behavior. To ensure compatibility, set `attn_implementation` in the model "
                    "configuration to one of these supported options or verify that your attention mechanism can "
                    "handle flattened sequences."
                )

            if args.per_device_train_batch_size == 1 and not args.packing:
                logger.warning(
                    "You are using a per_device_train_batch_size of 1 with padding-free training. Using a batch size "
                    "of 1 anihilate the benefits of padding-free training. Please consider increasing the batch size "
                    "to at least 2."
                )

        # Decide whether to use completion-only loss: if not specified, then it is set to True if the dataset format
        # is prompt-completion, and False if the dataset format is language modeling.
        dataset_sample = next(iter(train_dataset))
        if args.completion_only_loss is None:
            self.completion_only_loss = "prompt" in dataset_sample and "completion" in dataset_sample
        else:
            self.completion_only_loss = args.completion_only_loss

        self._is_vision_dataset = "image" in dataset_sample or "images" in dataset_sample
        if self._is_vision_dataset and not self._is_vlm:
            raise ValueError(
                "The dataset appears to be vision-related (contains 'image' or 'images' keys), but the provided "
                "model does not seem to be a vision-language model. Please check your model and dataset."
            )

        if data_collator is None and not self._is_vision_dataset:
            # Get the pad token: if not provided, use the one from the processing class or the eos token
            # if the processing class does not have a pad token.
            pad_token = args.pad_token or tokenizer.pad_token or tokenizer.eos_token
            pad_token_id = tokenizer.convert_tokens_to_ids(pad_token)
            if pad_token_id is None:
                raise ValueError(
                    f"The specified `pad_token` ('{pad_token}') is not found in the vocabulary of the given "
                    f"`processing_class` ({processing_class.__class__.__name__}). Ensure that the `pad_token` exists "
                    "in the vocabulary before using it as a padding token."
                )
            data_collator = DataCollatorForLanguageModeling(
                pad_token_id=pad_token_id,
                completion_only_loss=self.completion_only_loss,
                padding_free=self.padding_free,
                pad_to_multiple_of=args.pad_to_multiple_of,
            )
        elif data_collator is None and self._is_vision_dataset:
            data_collator = DataCollatorForVisionLanguageModeling(
                processor=processing_class,
                max_length=args.max_length,
                completion_only_loss=self.completion_only_loss,
                pad_to_multiple_of=args.pad_to_multiple_of,
                dataset_text_field=args.dataset_text_field,
            )

        if args.packing and args.packing_strategy == "bfd" and not use_flash_attention:
            logger.warning(
                "You are using packing, but the attention implementation is not set to a supported flash attention "
                "variant. Packing gathers multiple samples into a single sequence, and only the following "
                f"implementations are known to reliably support this: {', '.join(sorted(FLASH_ATTENTION_VARIANTS))}. "
                "Using other implementations may lead to cross-contamination between samples. To avoid this, either "
                "disable packing by setting `packing=False`, or set `attn_implementation` in the model configuration "
                "to one of these supported options."
            )
        if args.assistant_only_loss and not is_conversational(dataset_sample):
            raise ValueError(
                "You set `assistant_only_loss=True`, but the dataset is not conversational. This option is only "
                "supported for conversational datasets."
            )

        # Dataset
        # Skip dataset preparation if `skip_prepare_dataset=True` in `dataset_kwargs`, or if it's a VLM, where
        # preprocessing (e.g., image-to-pixel conversion) is too costly and done on the fly instead.
        skip_prepare_dataset = (
            args.dataset_kwargs is not None
            and args.dataset_kwargs.get("skip_prepare_dataset", False)
            or self._is_vision_dataset
        )
        if not skip_prepare_dataset:
            if self.completion_only_loss and formatting_func:
                raise ValueError(
                    "A formatting function was provided while `completion_only_loss=True`, which is incompatible. "
                    "Using a formatter converts the dataset to a language modeling type, conflicting with "
                    "completion-only loss. To resolve this, apply your formatting function before passing the "
                    "dataset, or disable `completion_only_loss` in `SFTConfig`."
                )
            train_dataset = self._prepare_dataset(
                train_dataset, processing_class, args, args.packing, formatting_func, "train"
            )
            if eval_dataset is not None:
                packing = args.packing if args.eval_packing is None else args.eval_packing
                if isinstance(eval_dataset, dict):
                    eval_dataset = {
                        key: self._prepare_dataset(dataset, processing_class, args, packing, formatting_func, key)
                        for key, dataset in eval_dataset.items()
                    }
                else:
                    eval_dataset = self._prepare_dataset(
                        eval_dataset, processing_class, args, packing, formatting_func, "eval"
                    )

        # Loss function
        if args.loss_type == "nll":
            pass  # use the default loss
        elif args.loss_type == "dft":
            if compute_loss_func is not None:
                raise ValueError(
                    "You passed a `compute_loss_func` together with `loss_type='dft'` to the `SFTTrainer`. "
                    "When using `loss_type='dft'`, the loss function is automatically set to the DFT loss, so passing a "
                    "`compute_loss_func` is not allowed."
                )
            compute_loss_func = dft_loss
        else:
            raise ValueError(f"Invalid `loss_type` {args.loss_type} passed. Supported values are 'nll' and 'dft'.")

        # Initialize the metrics
        self._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}
        self._total_train_tokens = 0

        # Initialize the Trainer. Parent class will handle:
        # - DeepSpeed configuration (through create_accelerator_and_postprocess)
        # - FSDP setup
        # - Distributed training setup
        # - Optimizer and scheduler creation

        super().__init__(
            model=model,
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
        )

        # Initialize activation offloading context
        if self.args.activation_offloading:
            self.maybe_activation_offload_context = get_act_offloading_ctx_manager(model=self.model)
        else:
            self.maybe_activation_offload_context = contextlib.nullcontext()

        # Add tags for models that have been loaded with the correct transformers version
        if hasattr(self.model, "add_model_tags"):
            self.model.add_model_tags(self._tag_names)

        self.aux_loss_enabled = getattr(model.config, "output_router_logits", False)

        self.teacher_model = None
        distill_weight = float(getattr(args, 'distill_weight', 0.0))

        if teacher_model is not None and distill_weight > 0:
            logger.info("Loading teacher model for knowledge distillation...")
            

            teacher_model_init_kwargs = getattr(args, 'teacher_model_init_kwargs', None) or {}
            

            if teacher_model_init_kwargs and not isinstance(teacher_model, str):
                logger.warning(
                    "You passed teacher_model_init_kwargs, but teacher_model is already instantiated. "
                    "The teacher_model_init_kwargs will be ignored."
                )
                teacher_model_init_kwargs = {}
            

            if 'dtype' in teacher_model_init_kwargs:
                dtype_value = teacher_model_init_kwargs['dtype']
                if dtype_value not in ["auto", None]:
                    teacher_model_init_kwargs['dtype'] = getattr(torch, dtype_value)
            
            if isinstance(teacher_model, str):
                logger.info(f"Loading teacher model from: {teacher_model}")
                if 'torch_dtype' not in teacher_model_init_kwargs:
                    teacher_model_init_kwargs['torch_dtype'] = torch.bfloat16
                teacher_model = AutoModelForCausalLM.from_pretrained(teacher_model, **teacher_model_init_kwargs)
            elif isinstance(teacher_model, (PreTrainedModel, nn.Module)):
                logger.info("Using provided teacher model instance")
            else:
                raise TypeError(
                    f"teacher_model must be a string or PreTrainedModel/nn.Module, got {type(teacher_model)}"
                )
            

            teacher_model.eval()
            for param in teacher_model.parameters():
                param.requires_grad = False
            
            if self.is_deepspeed_enabled:
                if hasattr(teacher_model, 'gradient_checkpointing_disable'):
                    teacher_model.gradient_checkpointing_disable()
                self.teacher_model = prepare_deepspeed(teacher_model, self.accelerator)
            else:
                self.teacher_model = self.accelerator.prepare_model(teacher_model, evaluation_mode=True)
            
            logger.info("Teacher model loaded successfully")
        elif teacher_model is not None and distill_weight == 0:
            logger.warning(
                "A teacher_model was provided but distill_weight is 0. The teacher model will not be used."
            )


    def _prepare_dataset(
        self,
        dataset: Union[Dataset, IterableDataset],
        processing_class: Union[PreTrainedTokenizerBase, BaseImageProcessor, FeatureExtractionMixin, ProcessorMixin],
        args: SFTConfig,
        packing: bool,
        formatting_func: Optional[Callable[[dict], str]],
        dataset_name: str,
    ) -> Union[Dataset, IterableDataset]:
        # Tabular backends like Arrow/Parquet insert `None` for mismatched keys in nested structures. Clean them from
        # sampled data.
        if isinstance(dataset, Dataset):  # IterableDataset does not support `with_transform`
            dataset = dataset.with_transform(remove_none_values)

        # If the dataset is already preprocessed (tokenized), skip the processing steps.
        column_names = get_dataset_column_names(dataset)
        is_processed = "input_ids" in column_names

        # Build the kwargs for the `map` function
        map_kwargs = {}
        if isinstance(dataset, Dataset):  # IterableDataset does not support num_proc
            map_kwargs["num_proc"] = args.dataset_num_proc

        with PartialState().main_process_first():
            # Apply the formatting function if any
            if formatting_func is not None and is_processed:
                logger.warning(
                    "You passed a dataset that is already processed (contains an `input_ids` field) together with a "
                    "formatting function. Therefore `formatting_func` will be ignored. Either remove the "
                    "`formatting_func` or pass a dataset that is not already processed.",
                )

            if formatting_func is not None and not is_processed:
                if isinstance(dataset, Dataset):  # `IterableDataset.map` does not support `desc`
                    map_kwargs["desc"] = f"Applying formatting function to {dataset_name} dataset"

                def _func(example):
                    return {"text": formatting_func(example)}

                dataset = dataset.map(_func, batched=False, **map_kwargs)

            if not is_processed:
                # Convert the dataset to ChatML if needed
                first_example = next(iter(dataset))
                if is_conversational_from_value(first_example):
                    if isinstance(dataset, Dataset):  # `IterableDataset.map` does not support `desc`
                        map_kwargs["desc"] = f"Converting {dataset_name} dataset to ChatML"
                    column_names = get_dataset_column_names(dataset)
                    dataset = dataset.map(
                        maybe_convert_to_chatml,
                        remove_columns="conversations" if "conversations" in column_names else None,
                        **map_kwargs,
                    )

                # Apply the chat template if needed
                first_example = next(iter(dataset))
                if not is_conversational(first_example):
                    if isinstance(dataset, Dataset):  # `IterableDataset.map` does not support `desc`
                        map_kwargs["desc"] = f"Adding EOS to {dataset_name} dataset"

                    def add_eos(example, eos_token):
                        if "text" in example and not example["text"].endswith(eos_token):  # language modeling case
                            example["text"] = example["text"] + eos_token
                        elif "completion" in example and not example["completion"].endswith(eos_token):
                            example["completion"] = example["completion"] + eos_token
                        return example

                    dataset = dataset.map(
                        add_eos,
                        fn_kwargs={"eos_token": processing_class.eos_token},
                        remove_columns="messages" if "messages" in column_names else None,  # renamed to "text"
                        **map_kwargs,
                    )

                # Tokenize the dataset
                if isinstance(dataset, Dataset):  # `IterableDataset.map` does not support `desc`
                    map_kwargs["desc"] = f"Tokenizing {dataset_name} dataset"

                def tokenize_fn(example, processing_class, dataset_text_field, assistant_only_loss):
                    if "prompt" in example:  # prompt-completion case
                        output = {}
                        if is_conversational(example):
                            if self._is_vlm:
                                prompt = prepare_multimodal_messages(example["prompt"], images=[])
                                completion = prepare_multimodal_messages(example["completion"], images=[])
                            else:
                                prompt = example["prompt"]
                                completion = example["completion"]
                            prompt_ids = processing_class.apply_chat_template(
                                prompt,
                                tokenize=True,
                                add_generation_prompt=True,
                                tools=example.get("tools"),
                                **example.get("chat_template_kwargs", {}),
                            )
                            # Fix transformers inconsistency: for VLMs, apply_chat_template returns lists of lists
                            # even for single examples, while for LLMs it returns lists of ints.
                            prompt_ids = prompt_ids[0] if isinstance(prompt_ids[0], list) else prompt_ids
                            prompt_completion_processed = processing_class.apply_chat_template(
                                prompt + completion,
                                return_dict=True,
                                tokenize=True,
                                return_assistant_tokens_mask=assistant_only_loss,
                                tools=example.get("tools"),
                                **example.get("chat_template_kwargs", {}),
                            )
                            # Fix transformers inconsistency: for VLMs, apply_chat_template returns lists of lists
                            # even for single examples, while for LLMs it returns lists of ints.
                            prompt_completion_processed = {
                                k: v[0] if isinstance(v[0], list) else v
                                for k, v in prompt_completion_processed.items()
                            }
                            prompt_completion_ids = prompt_completion_processed["input_ids"]
                            if "assistant_masks" in prompt_completion_processed:
                                output["assistant_masks"] = prompt_completion_processed["assistant_masks"]
                        else:
                            prompt_ids = processing_class(text=example["prompt"])["input_ids"]
                            prompt_completion_ids = processing_class(text=example["prompt"] + example["completion"])[
                                "input_ids"
                            ]

                        # Check if the tokenized prompt starts with the tokenized prompt+completion
                        if not prompt_completion_ids[: len(prompt_ids)] == prompt_ids:
                            logger.warning(
                                "Mismatch between tokenized prompt and the start of tokenized prompt+completion. "
                                "This may be due to unexpected tokenizer behavior, whitespace issues, or special "
                                "token handling. Verify that the tokenizer is processing text consistently."
                            )

                        # Create completion mask
                        completion_mask = [0] * len(prompt_ids) + [1] * (len(prompt_completion_ids) - len(prompt_ids))
                        output["input_ids"] = prompt_completion_ids
                        output["completion_mask"] = completion_mask

                    else:  # language modeling case
                        if is_conversational(example):
                            if self._is_vlm:
                                messages = prepare_multimodal_messages(example["messages"], images=[])
                            else:
                                messages = example["messages"]
                            processed = processing_class.apply_chat_template(
                                messages,
                                return_dict=True,
                                tokenize=True,
                                return_assistant_tokens_mask=assistant_only_loss,
                                tools=example.get("tools"),
                                **example.get("chat_template_kwargs", {}),
                            )
                            # Fix transformers inconsistency: for VLMs, apply_chat_template returns lists of lists
                            # even for single examples, while for LLMs it returns lists of ints.
                            processed = {k: v[0] if isinstance(v[0], list) else v for k, v in processed.items()}
                            output = {k: processed[k] for k in ("input_ids", "assistant_masks") if k in processed}
                        else:
                            output = {"input_ids": processing_class(text=example[dataset_text_field])["input_ids"]}

                    if "assistant_masks" in output and 1 not in output["assistant_masks"]:
                        raise RuntimeError(
                            "You're using `assistant_only_loss=True`, but at least one example has no assistant "
                            "tokens. This usually means the tokenizer's chat template doesn't generate assistant "
                            "masks — it may be missing the `{% generation %}` keyword. Please check the template and "
                            "ensure it's correctly configured to support assistant masking."
                        )
                    return output

                dataset = dataset.map(
                    tokenize_fn,
                    fn_kwargs={
                        "processing_class": processing_class,
                        "dataset_text_field": args.dataset_text_field,
                        "assistant_only_loss": args.assistant_only_loss,
                    },
                    **map_kwargs,
                )

            # Pack or truncate
            if packing:
                if args.max_length is None:
                    raise ValueError("When packing is enabled, `max_length` can't be `None`.")
                if isinstance(dataset, Dataset):  # `IterableDataset.map` does not support `desc`
                    map_kwargs["desc"] = f"Packing {dataset_name} dataset"

                columns = ["input_ids"]
                if "completion_mask" in get_dataset_column_names(dataset):
                    columns.append("completion_mask")
                if "assistant_masks" in get_dataset_column_names(dataset):
                    columns.append("assistant_masks")

                dataset = dataset.select_columns(columns)

                # Packing adds new column "seq_lengths" needed for document aware FlashAttention
                dataset = pack_dataset(dataset, args.max_length, args.packing_strategy, map_kwargs)
            elif args.max_length is not None:
                if isinstance(dataset, Dataset):  # `IterableDataset.map` does not support `desc`
                    map_kwargs["desc"] = f"Truncating {dataset_name} dataset"
                dataset = truncate_dataset(dataset, args.max_length, map_kwargs)
            # For Liger kernel, ensure only the essential columns
            if args.use_liger_kernel:
                collator_expected_keys = {"input_ids", "seq_lengths", "completion_mask", "assistant_masks"}
                column_names = get_dataset_column_names(dataset)
                dataset = dataset.select_columns(collator_expected_keys.intersection(column_names))

        return dataset

    def _set_signature_columns_if_needed(self):
        # If `self.args.remove_unused_columns` is True, non-signature columns are removed.
        # By default, this method sets `self._signature_columns` to the model's expected inputs (usually, "input_ids"
        # and "attention_mask"). When using `train_on_completion_only` we add a "completion_mask" column to the
        # dataset. So we need to override the default signature columns to include "completion_mask" as well.
        if self._signature_columns is None:
            if self._is_vision_dataset:
                self._signature_columns = ["messages", "prompt", "completion", "images"]
            else:
                self._signature_columns = ["input_ids", "labels", "seq_lengths", "completion_mask", "assistant_masks"]

    def compute_sft_loss(
        self,
        model: nn.Module,
        inputs: dict[str, Union[torch.Tensor, Any]],
        return_outputs: bool = False,
        num_items_in_batch: Optional[torch.Tensor] = None,
    ):
        """
        
        """

        outputs = model(**inputs)
        logits = outputs.logits
        labels = inputs["labels"]
        

        loss_fct = nn.CrossEntropyLoss(reduction='none', ignore_index=-100)
        

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        

        # shift_logits: [batch_size, seq_len-1, vocab_size]
        # shift_labels: [batch_size, seq_len-1]
        batch_size, seq_len, vocab_size = shift_logits.shape
        

        token_loss = loss_fct(
            shift_logits.view(-1, vocab_size),
            shift_labels.view(-1)
        )  # [batch_size * (seq_len-1)]
        

        token_loss = token_loss.view(batch_size, seq_len)
        

        mask = (shift_labels != -100).float()
        

        if num_items_in_batch is not None:
            loss = (token_loss * mask).sum() / num_items_in_batch
        else:
            loss = (token_loss * mask).sum() / mask.sum().clamp(min=1.0)
        

        outputs.token_loss = token_loss
        outputs.token_mask = mask
        

        return (loss, outputs) if return_outputs else loss

    def compute_random_label_sft_loss(
        self,
        model: nn.Module,
        inputs: dict[str, Union[torch.Tensor, Any]],
        return_outputs: bool = False,
        num_items_in_batch: Optional[torch.Tensor] = None,
    ):
        outputs = model(**inputs)
        logits = outputs.logits
        labels = inputs["labels"]

        loss_fct = nn.CrossEntropyLoss(reduction="none", ignore_index=-100)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        random_labels = torch.randint(
            low=0,
            high=shift_logits.size(-1),
            size=shift_labels.shape,
            device=shift_labels.device,
            dtype=shift_labels.dtype,
        )
        shift_labels = torch.where(shift_labels != -100, random_labels, shift_labels)

        batch_size, seq_len, vocab_size = shift_logits.shape
        token_loss = loss_fct(shift_logits.view(-1, vocab_size), shift_labels.view(-1))
        token_loss = token_loss.view(batch_size, seq_len)

        mask = (shift_labels != -100).float()
        if num_items_in_batch is not None:
            loss = (token_loss * mask).sum() / num_items_in_batch
        else:
            loss = (token_loss * mask).sum() / mask.sum().clamp(min=1.0)

        outputs.token_loss = token_loss
        outputs.token_mask = mask
        return (loss, outputs) if return_outputs else loss

    def _get_total_training_steps(self) -> int:
        total_steps = int(getattr(self.state, "max_steps", 0) or 0)
        if total_steps > 0:
            return total_steps

        gradient_accumulation_steps = max(int(getattr(self.args, "gradient_accumulation_steps", 1)), 1)
        num_train_epochs = max(float(getattr(self.args, "num_train_epochs", 1)), 1.0)
        try:
            steps_per_epoch = len(self.get_train_dataloader())
        except Exception:
            return 1

        return max(int(math.ceil(steps_per_epoch * num_train_epochs / gradient_accumulation_steps)), 1)

    def _get_temperature_schedule_value(self) -> float:
        phi_type = getattr(self.args, "phi_type", None)
        temperature_min = float(getattr(self.args, "temperature_min", 1.0))
        temperature_max = float(getattr(self.args, "temperature_max", 4.0))

        if phi_type not in TEMPERATURE_SCHEDULE_PHI_TYPES:
            raise ValueError(f"Unsupported temperature schedule phi_type: {phi_type}")

        total_steps = self._get_total_training_steps()
        current_step = int(getattr(self.state, "global_step", 0) or 0)
        current_step = min(max(current_step, 0), total_steps - 1)

        if total_steps == 1:
            return temperature_max if phi_type == "temperature_schedule_high_to_low" else temperature_min

        progress = current_step / float(total_steps - 1)
        if phi_type == "temperature_schedule_high_to_low":
            return temperature_max - (temperature_max - temperature_min) * progress
        return temperature_min + (temperature_max - temperature_min) * progress

    @staticmethod
    def _temperature_scheduled_kl_loss(
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor,
        temperature: float,
    ) -> torch.Tensor:
        if temperature <= 0:
            raise ValueError("temperature must be > 0.")

        if student_logits.size(-1) != teacher_logits.size(-1):
            common_vocab_size = min(student_logits.size(-1), teacher_logits.size(-1))
            student_logits = student_logits[..., :common_vocab_size]
            teacher_logits = teacher_logits[..., :common_vocab_size]

        sanitized_student_logits = torch.where(
            torch.isfinite(student_logits), student_logits, torch.full_like(student_logits, -1e9)
        )
        sanitized_teacher_logits = torch.where(
            torch.isfinite(teacher_logits), teacher_logits, torch.full_like(teacher_logits, -1e9)
        )

        scaled_student_log_probs = F.log_softmax(
            sanitized_student_logits / temperature, dim=-1, dtype=torch.float32
        )
        scaled_teacher_probs = F.softmax(sanitized_teacher_logits / temperature, dim=-1, dtype=torch.float32)

        token_kl = F.kl_div(scaled_student_log_probs, scaled_teacher_probs, reduction="none").sum(dim=-1)
        valid_mask = (labels != -100).float()
        normalizer = valid_mask.sum().clamp_min(1.0)

        return (temperature * temperature) * (token_kl * valid_mask).sum() / normalizer

    def compute_loss(
        self,
        model: nn.Module,
        inputs: dict[str, Union[torch.Tensor, Any]],
        return_outputs: bool = False,
        num_items_in_batch: Optional[torch.Tensor] = None,
    ):
        """
        Compute training loss and additionally compute token accuracies
        """
        mode = "train" if self.model.training else "eval"

        # Set aside labels as it will be dropped by super().compute_loss() if a custom `compute_loss_func` is used.
        # This can be removed when this issue is fixed.
        labels = inputs["labels"]

        # If not set, defaults from model config and may warn since cache isn't compatible with gradient checkpointing
        inputs["use_cache"] = False

        if self.args.phi_type == "sft":
            (loss, outputs) = super().compute_loss(
                model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch
            )
        elif self.args.phi_type in [
            "adap_sft",
            "reverse_adap_sft",
            "entropy_sft_distill",
            "entropy_sft_sft",
            "curriculum_weight",
            "reverse_curriculum_weight",
            "wsl",
            "reverse_wsl",
            "temperature_schedule_high_to_low",
            "temperature_schedule_low_to_high",
        ]:

            (loss, outputs) = self.compute_sft_loss(
                model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch
            )
        elif self.args.phi_type == "soft_kd_entropy_reg":
            (loss, outputs) = super().compute_loss(
                model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch
            )
        elif self.args.phi_type == "random_label":
            (loss, outputs) = self.compute_random_label_sft_loss(
                model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch
            )
        elif self.args.phi_type in ["chi2", "chi2_curriculum"]:
            (loss, outputs) = self._compute_kd_loss(
                model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch
            )
        else:
            raise ValueError(f"Invalid phi_type: {self.args.phi_type}")
        
        teacher_logits = None
        student_logits_shift = None
        teacher_logits_shift = None
        shift_labels = None
        distill_loss = None
    
        if self.teacher_model is not None:
            with torch.no_grad():
                teacher_outputs = self.teacher_model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    use_cache=False
                )
                teacher_logits = teacher_outputs.logits
                

                teacher_per_token_entropy = entropy_from_logits(teacher_logits)
                if (self.num_virtual_tokens > 0 and 
                    model.peft_config[model.active_adapter].peft_type != PeftType.PREFIX_TUNING):
                    teacher_per_token_entropy = teacher_per_token_entropy[:, self.num_virtual_tokens:]
                
                if "attention_mask" in inputs:
                    attention_mask = inputs["attention_mask"]
                    teacher_entropy = (teacher_per_token_entropy * attention_mask).sum() / attention_mask.sum()
                elif "position_ids" in inputs:
                    teacher_entropy = teacher_per_token_entropy.mean()
                else:
                    raise ValueError("Expected 'attention_mask' or 'position_ids' in inputs.")
                
                teacher_entropy = self.accelerator.gather_for_metrics(teacher_entropy).mean().item()
                self._metrics[mode]["teacher_entropy"].append(teacher_entropy)
                del teacher_per_token_entropy


                shift_labels = labels[:, 1:].contiguous()  # [B, T-1]
                student_logits_shift = outputs.logits[:, :-1, :].contiguous()  # [B, T-1, V]
                teacher_logits_shift = teacher_logits[:, :-1, :].contiguous()  # [B, T-1, V]
                
                vocab_size = student_logits_shift.size(-1)
                safe_labels = shift_labels.clamp(0, vocab_size - 1)
            if self.args.phi_type == "adap_sft" or "adap" in self.args.phi_type:

                student_vocab_size = student_logits_shift.size(-1)
                teacher_vocab_size = teacher_logits_shift.size(-1)
                vocab_size = min(student_vocab_size, teacher_vocab_size)
                

                valid_label_mask = (shift_labels >= 0) & (shift_labels < vocab_size)
                safe_labels = torch.where(valid_label_mask, shift_labels, torch.zeros_like(shift_labels))
                

                ll_mask = valid_label_mask.float()
                

                student_log_probs = F.log_softmax(student_logits_shift[..., :vocab_size], dim=-1)
                student_token_ll = torch.gather(
                    student_log_probs, -1, safe_labels.unsqueeze(-1)
                ).squeeze(-1)  # [B, T-1]
                
                teacher_log_probs = F.log_softmax(teacher_logits_shift[..., :vocab_size], dim=-1)
                teacher_token_ll = torch.gather(
                    teacher_log_probs, -1, safe_labels.unsqueeze(-1)
                ).squeeze(-1)  # [B, T-1]


                student_token_ll = student_token_ll * ll_mask
                teacher_token_ll = teacher_token_ll * ll_mask                
            

            if not self.args.use_liger_kernel:
                student_logits_shift = outputs.logits[:, :-1, :].contiguous()
                teacher_logits_shift = teacher_logits[:, :-1, :].contiguous()
                del teacher_logits
                labels_shift = labels[:, 1:].contiguous()
                if self.args.phi_type in TEMPERATURE_SCHEDULE_PHI_TYPES:
                    current_temperature = self._get_temperature_schedule_value()
                    distill_loss = self._temperature_scheduled_kl_loss(
                        student_logits=student_logits_shift,
                        teacher_logits=teacher_logits_shift,
                        labels=labels_shift,
                        temperature=current_temperature,
                    )
                    self._metrics[mode]["scheduled_temperature"].append(current_temperature)
                elif self.args.div == "ab":
                    distill_loss = self.ab_div_loss(
                        student_logits=student_logits_shift,
                        teacher_logits=teacher_logits_shift,
                        labels=labels_shift,
                        alpha=getattr(self.args, 'distill_alpha', 1.0),
                        beta=getattr(self.args, 'distill_beta', 0.0),
                    )
                elif self.args.div == "sfkl":
                    distill_loss = self.skewed_forward_kl(
                        logits=student_logits_shift,
                        teacher_logits=teacher_logits_shift,
                        no_model_batch=labels_shift,
                    )
                elif self.args.div == "srkl":
                    distill_loss = self.skewed_reverse_kl(
                        logits=student_logits_shift,
                        teacher_logits=teacher_logits_shift,
                        no_model_batch=labels_shift,
                    )
                elif self.args.div == "tv":
                    distill_loss = self.tv_distance(
                        logits=student_logits_shift,
                        teacher_logits=teacher_logits_shift,
                        no_model_batch=labels_shift,
                    )
                elif self.args.div == "akl":
                    distill_loss = self.akd_loss(
                        logits=student_logits_shift,
                        teacher_logits=teacher_logits_shift,
                        no_model_batch=labels_shift,
                        mu=0.5
                    )
                elif self.args.div == "jsd":
                    distill_loss = self.js_distance(
                        logits=student_logits_shift,
                        teacher_logits=teacher_logits_shift,
                        no_model_batch=labels_shift,
                        lam=0.1
                    )
                else:
                    raise ValueError("Invalid div type.")

                self._metrics[mode]["sft_loss"].append(self._safe_metric_value(loss))
                self._metrics[mode]["distill_loss"].append(self._safe_metric_value(distill_loss))
                

                if self.args.phi_type == "adap_sft" and hasattr(outputs, 'token_loss'):

                    with torch.no_grad():
                        teacher_probs = F.softmax(teacher_logits_shift, dim=-1, dtype=torch.float32)
                        

                        vocab_size = teacher_probs.size(-1)
                        valid_label_mask = (labels_shift >= 0) & (labels_shift < vocab_size)
                        safe_labels = torch.where(valid_label_mask, labels_shift, torch.zeros_like(labels_shift))
                        
                        teacher_token_probs = torch.gather(
                            teacher_probs, -1, safe_labels.unsqueeze(-1)
                        ).squeeze(-1)  # [batch_size, seq_len-1]
                        

                        teacher_token_probs = torch.where(valid_label_mask, teacher_token_probs, torch.zeros_like(teacher_token_probs))
                    

                    token_loss = outputs.token_loss
                    loss_mask = outputs.token_mask
                    weighted_sft_loss = teacher_token_probs * token_loss * loss_mask
                    sft_loss_weighted = weighted_sft_loss.sum() / (num_items_in_batch if num_items_in_batch is not None else loss_mask.sum().clamp(min=1.0))
                    

                    distill_weight_per_token = 1.0 - teacher_token_probs
                    distill_loss_weighted = self.ab_div_loss(
                        student_logits=student_logits_shift,
                        teacher_logits=teacher_logits_shift,
                        labels=labels_shift,
                        alpha=getattr(self.args, 'distill_alpha', 1.0),
                        beta=getattr(self.args, 'distill_beta', 0.0),
                        weight=distill_weight_per_token,
                    )
                    
                    loss = sft_loss_weighted + distill_loss_weighted
                
                elif self.args.phi_type == "entropy_sft_distill" and hasattr(outputs, 'token_loss'):


                    with torch.no_grad():
                        teacher_probs = F.softmax(teacher_logits_shift, dim=-1, dtype=torch.float32)
                        


                        gamma = getattr(self.args, 'entropy_gamma', 0.5)
                        power_sum = torch.sum(teacher_probs ** gamma, dim=-1)  # [batch_size, seq_len-1]
                        


                        vocab_size = teacher_probs.size(-1)
                        min_power_sum = 1.0
                        max_power_sum = vocab_size ** (1 - gamma)
                        normalized_smoothness = (power_sum - min_power_sum) / (max_power_sum - min_power_sum)
                        normalized_smoothness = normalized_smoothness.clamp(0.0, 1.0)
                        

                        valid_label_mask = (labels_shift >= 0) & (labels_shift < vocab_size)

                        normalized_smoothness = torch.where(valid_label_mask, normalized_smoothness, torch.zeros_like(normalized_smoothness))
                    

                    token_loss = outputs.token_loss
                    loss_mask = outputs.token_mask
                    sft_weight_per_token = 1.0 - normalized_smoothness
                    weighted_sft_loss = sft_weight_per_token * token_loss * loss_mask
                    sft_loss_weighted = weighted_sft_loss.sum() / (num_items_in_batch if num_items_in_batch is not None else loss_mask.sum().clamp(min=1.0))
                    

                    distill_loss_weighted = self.ab_div_loss(
                        student_logits=student_logits_shift,
                        teacher_logits=teacher_logits_shift,
                        labels=labels_shift,
                        alpha=getattr(self.args, 'distill_alpha', 1.0),
                        beta=getattr(self.args, 'distill_beta', 0.0),
                        weight=normalized_smoothness,
                    )
                    
                    loss = sft_loss_weighted + distill_loss_weighted
                
                elif self.args.phi_type == "entropy_sft_sft" and hasattr(outputs, 'token_loss'):


                    with torch.no_grad():
                        teacher_probs = F.softmax(teacher_logits_shift, dim=-1, dtype=torch.float32)
                        

                        gamma = getattr(self.args, 'entropy_gamma', 0.5)
                        power_sum = torch.sum(teacher_probs ** gamma, dim=-1)  # [batch_size, seq_len-1]
                        

                        vocab_size = teacher_probs.size(-1)
                        min_power_sum = 1.0
                        max_power_sum = vocab_size ** (1 - gamma)
                        normalized_smoothness = (power_sum - min_power_sum) / (max_power_sum - min_power_sum)
                        normalized_smoothness = normalized_smoothness.clamp(0.0, 1.0)
                        

                        valid_label_mask = (labels_shift >= 0) & (labels_shift < vocab_size)

                        normalized_smoothness = torch.where(valid_label_mask, normalized_smoothness, torch.zeros_like(normalized_smoothness))
                    

                    token_loss = outputs.token_loss
                    loss_mask = outputs.token_mask
                    weighted_sft_loss = normalized_smoothness * token_loss * loss_mask
                    sft_loss_weighted = weighted_sft_loss.sum() / (num_items_in_batch if num_items_in_batch is not None else loss_mask.sum().clamp(min=1.0))
                    

                    distill_weight_per_token = 1.0 - normalized_smoothness
                    distill_loss_weighted = self.ab_div_loss(
                        student_logits=student_logits_shift,
                        teacher_logits=teacher_logits_shift,
                        labels=labels_shift,
                        alpha=getattr(self.args, 'distill_alpha', 1.0),
                        beta=getattr(self.args, 'distill_beta', 0.0),
                        weight=distill_weight_per_token,
                    )
                    
                    loss = sft_loss_weighted + distill_loss_weighted
                
                elif self.args.phi_type == "curriculum_weight" and hasattr(outputs, 'token_loss'):


                    base_distill_weight = getattr(self.args, 'distill_weight', 0.5)
                    warmup_ratio = getattr(self.args, 'curriculum_warmup_ratio', 1.0)
                    

                    current_step = self.state.global_step

                    total_steps = self.state.max_steps if self.state.max_steps > 0 else (
                        len(self.get_train_dataloader()) * self.args.num_train_epochs // self.args.gradient_accumulation_steps
                    )
                    

                    warmup_steps = int(total_steps * warmup_ratio)
                    

                    if current_step < warmup_steps:

                        progress = current_step / max(warmup_steps, 1)
                        current_distill_weight = progress * base_distill_weight
                    else:

                        current_distill_weight = base_distill_weight
                    

                    loss = (1.0 - current_distill_weight) * loss + current_distill_weight * distill_loss
                    

                    if hasattr(self, '_metrics') and 'train' in self._metrics:
                        if 'curriculum_distill_weight' not in self._metrics['train']:
                            self._metrics['train']['curriculum_distill_weight'] = []
                        self._metrics['train']['curriculum_distill_weight'].append(current_distill_weight)
                
                elif self.args.phi_type == "reverse_curriculum_weight" and hasattr(outputs, 'token_loss'):



                    base_distill_weight = getattr(self.args, 'distill_weight', 0.5)
                    warmup_ratio = getattr(self.args, 'curriculum_warmup_ratio', 1.0)

                    current_step = self.state.global_step
                    total_steps = self.state.max_steps if self.state.max_steps > 0 else (
                        len(self.get_train_dataloader()) * self.args.num_train_epochs
                        // self.args.gradient_accumulation_steps
                    )

                    warmup_steps = int(total_steps * warmup_ratio)

                    if current_step < warmup_steps:
                        progress = current_step / max(warmup_steps, 1)
                        current_distill_weight = 1.0 - progress * (1.0 - base_distill_weight)
                    else:
                        current_distill_weight = base_distill_weight

                    loss = (1.0 - current_distill_weight) * loss + current_distill_weight * distill_loss


                    if hasattr(self, '_metrics') and 'train' in self._metrics:
                        if 'curriculum_distill_weight' not in self._metrics['train']:
                            self._metrics['train']['curriculum_distill_weight'] = []
                        self._metrics['train']['curriculum_distill_weight'].append(current_distill_weight)
                
                elif self.args.phi_type == "wsl" and hasattr(outputs, 'token_loss'):





                    
                    with torch.no_grad():

                        student_probs = F.softmax(student_logits_shift, dim=-1, dtype=torch.float32)
                        teacher_probs = F.softmax(teacher_logits_shift, dim=-1, dtype=torch.float32)
                        

                        vocab_size = student_probs.size(-1)
                        valid_label_mask = (labels_shift >= 0) & (labels_shift < vocab_size)
                        safe_labels = torch.where(valid_label_mask, labels_shift, torch.zeros_like(labels_shift))
                        

                        student_token_probs = torch.gather(
                            student_probs, -1, safe_labels.unsqueeze(-1)
                        ).squeeze(-1)  # [batch_size, seq_len-1]
                        
                        teacher_token_probs = torch.gather(
                            teacher_probs, -1, safe_labels.unsqueeze(-1)
                        ).squeeze(-1)  # [batch_size, seq_len-1]
                        

                        eps = 1e-8
                        student_token_probs = torch.where(
                            valid_label_mask, 
                            student_token_probs.clamp(min=eps), 
                            torch.ones_like(student_token_probs) * eps
                        )
                        teacher_token_probs = torch.where(
                            valid_label_mask, 
                            teacher_token_probs.clamp(min=eps), 
                            torch.ones_like(teacher_token_probs) * eps
                        )
                        




                        log_student = torch.log(student_token_probs)
                        log_teacher = torch.log(teacher_token_probs)
                        

                        log_teacher_safe = torch.where(
                            torch.abs(log_teacher) < eps,
                            torch.ones_like(log_teacher) * (-eps),
                            log_teacher
                        )
                        


                        ratio = log_student / log_teacher_safe
                        


                        ratio_clamped = ratio.clamp(min=-10.0, max=10.0)
                        wsl_weight = 1.0 - torch.exp(-ratio_clamped)
                        

                        wsl_weight = wsl_weight.clamp(min=0.0, max=1.0)
                        

                        wsl_weight = torch.where(valid_label_mask, wsl_weight, torch.zeros_like(wsl_weight))
                        

                        wsl_weight = wsl_weight.detach()
                    

                    distill_loss_weighted = self.ab_div_loss(
                        student_logits=student_logits_shift,
                        teacher_logits=teacher_logits_shift,
                        labels=labels_shift,
                        alpha=getattr(self.args, 'distill_alpha', 1.0),
                        beta=getattr(self.args, 'distill_beta', 0.0),
                        weight=wsl_weight,
                    )
                    


                    wsl_alpha = getattr(self.args, 'distill_weight', 0.5)
                    loss = loss + wsl_alpha * distill_loss_weighted
                    

                    if hasattr(self, '_metrics') and mode in self._metrics:
                        if 'wsl_weight_mean' not in self._metrics[mode]:
                            self._metrics[mode]['wsl_weight_mean'] = []
                        loss_mask = outputs.token_mask if hasattr(outputs, 'token_mask') else valid_label_mask.float()
                        wsl_weight_mean = (wsl_weight * loss_mask).sum() / loss_mask.sum().clamp(min=1.0)
                        self._metrics[mode]['wsl_weight_mean'].append(wsl_weight_mean.item())
                
                elif self.args.phi_type == "reverse_wsl" and hasattr(outputs, 'token_loss'):





                    
                    with torch.no_grad():

                        student_probs = F.softmax(student_logits_shift, dim=-1, dtype=torch.float32)
                        teacher_probs = F.softmax(teacher_logits_shift, dim=-1, dtype=torch.float32)
                        

                        vocab_size = student_probs.size(-1)
                        valid_label_mask = (labels_shift >= 0) & (labels_shift < vocab_size)
                        safe_labels = torch.where(valid_label_mask, labels_shift, torch.zeros_like(labels_shift))
                        

                        student_token_probs = torch.gather(
                            student_probs, -1, safe_labels.unsqueeze(-1)
                        ).squeeze(-1)  # [batch_size, seq_len-1]
                        
                        teacher_token_probs = torch.gather(
                            teacher_probs, -1, safe_labels.unsqueeze(-1)
                        ).squeeze(-1)  # [batch_size, seq_len-1]
                        

                        eps = 1e-8
                        student_token_probs = torch.where(
                            valid_label_mask, 
                            student_token_probs.clamp(min=eps), 
                            torch.ones_like(student_token_probs) * eps
                        )
                        teacher_token_probs = torch.where(
                            valid_label_mask, 
                            teacher_token_probs.clamp(min=eps), 
                            torch.ones_like(teacher_token_probs) * eps
                        )
                        

                        log_student = torch.log(student_token_probs)
                        log_teacher = torch.log(teacher_token_probs)
                        

                        log_teacher_safe = torch.where(
                            torch.abs(log_teacher) < eps,
                            torch.ones_like(log_teacher) * (-eps),
                            log_teacher
                        )
                        
                        ratio = log_student / log_teacher_safe
                        ratio_clamped = ratio.clamp(min=-10.0, max=10.0)
                        



                        reverse_wsl_weight = torch.exp(-ratio_clamped)
                        

                        reverse_wsl_weight = reverse_wsl_weight.clamp(min=0.0, max=1.0)
                        

                        reverse_wsl_weight = torch.where(valid_label_mask, reverse_wsl_weight, torch.zeros_like(reverse_wsl_weight))
                        

                        reverse_wsl_weight = reverse_wsl_weight.detach()
                    

                    distill_loss_weighted = self.ab_div_loss(
                        student_logits=student_logits_shift,
                        teacher_logits=teacher_logits_shift,
                        labels=labels_shift,
                        alpha=getattr(self.args, 'distill_alpha', 1.0),
                        beta=getattr(self.args, 'distill_beta', 0.0),
                        weight=reverse_wsl_weight,
                    )
                    

                    wsl_alpha = getattr(self.args, 'distill_weight', 0.5)
                    loss = loss + wsl_alpha * distill_loss_weighted
                    
                    if hasattr(self, '_metrics') and mode in self._metrics:
                        if 'reverse_wsl_weight_mean' not in self._metrics[mode]:
                            self._metrics[mode]['reverse_wsl_weight_mean'] = []
                        loss_mask = outputs.token_mask if hasattr(outputs, 'token_mask') else valid_label_mask.float()
                        reverse_wsl_weight_mean = (reverse_wsl_weight * loss_mask).sum() / loss_mask.sum().clamp(min=1.0)
                        self._metrics[mode]['reverse_wsl_weight_mean'].append(reverse_wsl_weight_mean.item())

                elif self.args.phi_type == "reverse_adap_sft" and hasattr(outputs, 'token_loss'):
                    with torch.no_grad():
                        teacher_probs = F.softmax(teacher_logits_shift, dim=-1, dtype=torch.float32)

                        vocab_size = teacher_probs.size(-1)
                        valid_label_mask = (labels_shift >= 0) & (labels_shift < vocab_size)
                        safe_labels = torch.where(valid_label_mask, labels_shift, torch.zeros_like(labels_shift))

                        teacher_token_probs = torch.gather(
                            teacher_probs, -1, safe_labels.unsqueeze(-1)
                        ).squeeze(-1)

                        teacher_token_probs = torch.where(valid_label_mask, teacher_token_probs, torch.zeros_like(teacher_token_probs))
                        teacher_token_probs = teacher_token_probs.detach()

                    token_loss = outputs.token_loss
                    loss_mask = outputs.token_mask
                    sft_weight_per_token = 1.0 - teacher_token_probs
                    weighted_sft_loss = sft_weight_per_token * token_loss * loss_mask
                    sft_loss_weighted = weighted_sft_loss.sum() / (num_items_in_batch if num_items_in_batch is not None else loss_mask.sum().clamp(min=1.0))

                    distill_loss_weighted = self.ab_div_loss(
                        student_logits=student_logits_shift,
                        teacher_logits=teacher_logits_shift,
                        labels=labels_shift,
                        alpha=getattr(self.args, 'distill_alpha', 1.0),
                        beta=getattr(self.args, 'distill_beta', 0.0),
                        weight=teacher_token_probs,
                    )

                    loss = sft_loss_weighted + distill_loss_weighted

                elif self.args.phi_type == "chi2_curriculum":

                    base_distill_weight = getattr(self.args, 'distill_weight', 0.5)
                    warmup_ratio = getattr(self.args, 'curriculum_warmup_ratio', 1.0)

                    current_step = self.state.global_step
                    total_steps = self.state.max_steps if self.state.max_steps > 0 else (
                        len(self.get_train_dataloader()) * self.args.num_train_epochs
                        // self.args.gradient_accumulation_steps
                    )

                    warmup_steps = int(total_steps * warmup_ratio)

                    if current_step < warmup_steps:
                        progress = current_step / max(warmup_steps, 1)
                        current_distill_weight = progress * base_distill_weight
                    else:
                        current_distill_weight = base_distill_weight

                    loss = (1.0 - current_distill_weight) * loss + current_distill_weight * distill_loss


                    if hasattr(self, '_metrics') and mode in self._metrics:
                        if 'curriculum_distill_weight' not in self._metrics[mode]:
                            self._metrics[mode]['curriculum_distill_weight'] = []
                        self._metrics[mode]['curriculum_distill_weight'].append(current_distill_weight)

                elif self.args.phi_type == "soft_kd_entropy_reg":
                    distill_weight = getattr(self.args, 'distill_weight', 0.5)
                    entropy_reg_weight = getattr(self.args, 'entropy_reg_weight', 0.1)
                    valid_mask = (labels_shift != -100).float()
                    student_entropy_reg = entropy_from_logits(student_logits_shift)
                    student_entropy_reg = (student_entropy_reg * valid_mask).sum() / valid_mask.sum().clamp(min=1.0)
                    loss = (1.0 - distill_weight) * loss + distill_weight * distill_loss - entropy_reg_weight * student_entropy_reg

                    if hasattr(self, '_metrics') and mode in self._metrics:
                        if 'student_entropy_reg' not in self._metrics[mode]:
                            self._metrics[mode]['student_entropy_reg'] = []
                        if 'entropy_reg_weight' not in self._metrics[mode]:
                            self._metrics[mode]['entropy_reg_weight'] = []
                        self._metrics[mode]['student_entropy_reg'].append(self._safe_metric_value(student_entropy_reg))
                        self._metrics[mode]['entropy_reg_weight'].append(entropy_reg_weight)

                else:

                    distill_weight = getattr(self.args, 'distill_weight', 0.5)
                    loss = (1.0 - distill_weight) * loss + distill_weight * distill_loss
                

                del teacher_logits_shift
                

        if isinstance(loss, torch.Tensor) and not torch.isfinite(loss).item():
            logger.warning("Non-finite total loss detected in compute_loss; sanitizing this step loss to zero.")
            loss = torch.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0)

        torch.cuda.empty_cache()
        # Compute entropy
        if not self.args.use_liger_kernel:  # liger doesn't return logits
            with torch.no_grad():
                logits_for_metrics = outputs.logits
                nonfinite_logits_count = (~torch.isfinite(logits_for_metrics)).sum()
                if nonfinite_logits_count.item() > 0:
                    logger.warning(
                        f"Found {nonfinite_logits_count.item()} non-finite logits while computing entropy metrics; "
                        "using sanitized logits for metric computation."
                    )
                safe_logits_for_metrics = torch.nan_to_num(logits_for_metrics, nan=0.0, posinf=0.0, neginf=0.0)
                per_token_entropy = entropy_from_logits(safe_logits_for_metrics)
                # When using Prompt Tuning, skip the virtual tokens in logits before entropy computation, since they
                # do not correspond to actual input tokens.
                if (
                    self.num_virtual_tokens > 0
                    and model.peft_config[model.active_adapter].peft_type != PeftType.PREFIX_TUNING
                ):
                    per_token_entropy = per_token_entropy[:, self.num_virtual_tokens :]
                if "attention_mask" in inputs:
                    attention_mask = inputs["attention_mask"]
                    entropy = torch.sum(per_token_entropy * attention_mask) / attention_mask.sum()
                elif "position_ids" in inputs:
                    entropy = torch.mean(per_token_entropy)
                else:
                    raise ValueError("Expected 'attention_mask' or 'position_ids' in inputs.")
                entropy = self.accelerator.gather_for_metrics(entropy).mean().item()
            self._metrics[mode]["entropy"].append(entropy)

        if mode == "train":
            # When using padding-free, the attention_mask is not present in the inputs, instead we have cu_seq_lens_q,
            # cu_seq_lens_k, and max_length_k, max_length_q and position_ids.
            if "attention_mask" in inputs:
                num_tokens_in_batch = self.accelerator.gather_for_metrics(inputs["attention_mask"].sum()).sum().item()
            elif "position_ids" in inputs:
                local_num_tokens = torch.tensor(inputs["position_ids"].size(1), device=inputs["position_ids"].device)
                num_tokens_in_batch = self.accelerator.gather_for_metrics(local_num_tokens).sum().item()
            else:
                raise ValueError("Expected 'attention_mask' or 'position_ids' in inputs.")
            self._total_train_tokens += num_tokens_in_batch
        self._metrics[mode]["num_tokens"] = [self._total_train_tokens]

        # Compute token accuracy if we have labels and if the model is not using Liger (no logits)
        if not self.args.use_liger_kernel:
            with torch.no_grad():
                if "shift_labels" in inputs:
                    # When using CP, labels are pre-shifted. We must use these (and cannot manually shift) because:
                    # - The first discarded token from inputs["labels"] actually belongs to process n-1
                    # - The last logits require the label from process n+1
                    shift_logits = outputs.logits.contiguous()
                    shift_labels = inputs["shift_labels"]
                else:
                    shift_logits = outputs.logits[..., :-1, :].contiguous()
                    shift_labels = labels[..., 1:].contiguous()

                # Prompt Tuning and P-Tuning output logits for virtual tokens but Prefix-Tuning does not.
                if (
                    self.num_virtual_tokens > 0
                    and model.peft_config[model.active_adapter].peft_type != PeftType.PREFIX_TUNING
                ):
                    shift_logits = shift_logits[:, self.num_virtual_tokens :, :]

                # Get predictions
                predictions = shift_logits.argmax(dim=-1)

                # Create mask for non-padding tokens (assuming ignore_index is -100)
                mask = shift_labels != -100

                # Calculate accuracy only on non-padding tokens
                correct_predictions = (predictions == shift_labels) & mask
                total_tokens = mask.sum()
                correct_tokens = correct_predictions.sum()

                # Gather the correct_tokens and total_tokens across all processes
                correct_tokens = self.accelerator.gather_for_metrics(correct_tokens)
                total_tokens = self.accelerator.gather_for_metrics(total_tokens)

                # Compute the mean token accuracy and log it
                total_sum = total_tokens.sum()
                accuracy = (correct_tokens.sum() / total_sum).item() if total_sum > 0 else 0.0
                self._metrics[mode]["mean_token_accuracy"].append(accuracy)
                if self.aux_loss_enabled:
                    aux_loss = outputs.aux_loss
                    aux_loss = self.accelerator.gather_for_metrics(aux_loss).mean().item()
                    self._metrics[mode]["aux_loss"].append(aux_loss)

        return (loss, outputs) if return_outputs else loss

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
        del student_logits, teacher_logits

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
        del log_p, log_q

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
        del safe_divergence, decay_weights

        if weight is not None:
            if weight.dim() > 1:
                weight = weight.squeeze()
            weighted_sum = (weight * token_losses).sum()
            sum_of_weights = torch.clamp(mask.sum(), min=eps)
            loss = weighted_sum / sum_of_weights
        else:
            loss = token_losses.sum() / torch.clamp(mask.sum(), min=1.0)

        return loss

    @staticmethod
    def tv_distance(logits, teacher_logits, no_model_batch):
        if logits.size(-1) != teacher_logits.size(-1):
            common_vocab_size = min(logits.size(-1), teacher_logits.size(-1))
            logits = logits[..., :common_vocab_size]
            teacher_logits = teacher_logits[..., :common_vocab_size]
        teacher_probs = F.softmax(teacher_logits, dim=-1, dtype=torch.float32)
        student_probs = F.softmax(logits, dim=-1, dtype=torch.float32)

        mask = (no_model_batch != -100).int()
        inf_mask = torch.isinf(logits) | torch.isinf(teacher_logits)
        prod_probs = 0.5 * torch.masked_fill(torch.abs(teacher_probs - student_probs), inf_mask, 0)
        x = torch.sum(prod_probs, dim=-1).view(-1)
        distil_loss = torch.sum(x * mask.view(-1), dim=0) / torch.sum(mask.view(-1), dim=0)
        return distil_loss

    @staticmethod
    def get_ratio(teacher_logits, logits, mu=0.5):
        teacher_probs = F.softmax(teacher_logits, dim=-1, dtype=torch.float32)
        student_probs = F.softmax(logits, dim=-1, dtype=torch.float32)

        teacher_entropy = -(teacher_probs * torch.log(teacher_probs + 1e-10)).sum(dim=-1)
        student_entropy = -(student_probs * torch.log(student_probs + 1e-10)).sum(dim=-1)

        s1 = torch.exp(-mu * teacher_entropy)
        s2 = torch.exp(-mu * student_entropy)
        return s1 / (s1 + s2), s2 / (s1 + s2)

    @staticmethod
    def get_kl(logits1, logits2, inf_mask, mask, ratio=None):
        probs1 = F.softmax(logits1, dim=-1, dtype=torch.float32)
        logprobs1 = F.log_softmax(logits1, dim=-1, dtype=torch.float32)
        prod_probs1 = torch.masked_fill(probs1 * logprobs1, inf_mask, 0)
        x1 = torch.sum(prod_probs1, dim=-1).view(-1)

        logprobs2 = F.log_softmax(logits2, dim=-1, dtype=torch.float32)
        prod_probs2 = torch.masked_fill(probs1 * logprobs2, inf_mask, 0)
        x2 = torch.sum(prod_probs2, dim=-1).view(-1)

        if ratio is None:
            distil_loss = torch.sum((x1 - x2) * mask.view(-1), dim=0) / torch.sum(mask.view(-1), dim=0)
        else:
            distil_loss = torch.sum((x1 - x2) * ratio.view(-1) * mask.view(-1), dim=0) / torch.sum(mask.view(-1), dim=0)
        return distil_loss

    @staticmethod
    def akd_loss(logits, teacher_logits, no_model_batch, mu=0.5):
        if logits.size(-1) != teacher_logits.size(-1):
            common_vocab_size = min(logits.size(-1), teacher_logits.size(-1))
            logits = logits[..., :common_vocab_size]
            teacher_logits = teacher_logits[..., :common_vocab_size]

        inf_mask = torch.isinf(logits) | torch.isinf(teacher_logits)
        mask = (no_model_batch != -100).int()

        h_ratio, l_ratio = SFTTrainer.get_ratio(teacher_logits, logits, mu)
        distil_loss = SFTTrainer.get_kl(teacher_logits, logits, inf_mask, mask, h_ratio) + \
                      SFTTrainer.get_kl(logits, teacher_logits, inf_mask, mask, l_ratio)
        return distil_loss

    @staticmethod
    def js_distance(logits, teacher_logits, no_model_batch, lam=0.1):
        if logits.size(-1) != teacher_logits.size(-1):
            common_vocab_size = min(logits.size(-1), teacher_logits.size(-1))
            logits = logits[..., :common_vocab_size]
            teacher_logits = teacher_logits[..., :common_vocab_size]

        teacher_probs = F.softmax(teacher_logits, dim=-1, dtype=torch.float32)
        student_probs = F.softmax(logits, dim=-1, dtype=torch.float32)
        mixed_probs = (1 - lam) * teacher_probs + lam * student_probs

        teacher_logprobs = F.log_softmax(teacher_logits, dim=-1, dtype=torch.float32)
        student_logprobs = F.log_softmax(logits, dim=-1, dtype=torch.float32)
        mixed_logprobs = torch.log(mixed_probs)

        mask = (no_model_batch != -100).int()
        inf_mask = torch.isinf(logits) | torch.isinf(teacher_logits)

        prod_probs = torch.masked_fill(student_probs * mixed_logprobs, inf_mask, 0)
        prod_probs -= torch.masked_fill(student_probs * student_logprobs, inf_mask, 0)
        x = torch.sum(prod_probs, dim=-1).view(-1)
        distil_loss = lam * -torch.sum(x * mask.view(-1), dim=0) / torch.sum(mask.view(-1), dim=0)

        prod_probs = torch.masked_fill(teacher_probs * mixed_logprobs, inf_mask, 0)
        prod_probs -= torch.masked_fill(teacher_probs * teacher_logprobs, inf_mask, 0)
        x = torch.sum(prod_probs, dim=-1).view(-1)
        distil_loss += (1 - lam) * -torch.sum(x * mask.view(-1), dim=0) / torch.sum(mask.view(-1), dim=0)

        return distil_loss

    @staticmethod
    def skewed_forward_kl(logits, teacher_logits, no_model_batch, lam=0.1):
        if logits.size(-1) != teacher_logits.size(-1):
            common_vocab_size = min(logits.size(-1), teacher_logits.size(-1))
            logits = logits[..., :common_vocab_size]
            teacher_logits = teacher_logits[..., :common_vocab_size]
        eps = torch.finfo(torch.float32).eps
        sanitized_logits = torch.where(torch.isfinite(logits), logits, torch.full_like(logits, -1e9))
        sanitized_teacher_logits = torch.where(torch.isfinite(teacher_logits), teacher_logits, torch.full_like(teacher_logits, -1e9))

        teacher_probs = F.softmax(sanitized_teacher_logits, dim=-1, dtype=torch.float32)
        student_probs = F.softmax(sanitized_logits, dim=-1, dtype=torch.float32)
        mixed_probs = lam * teacher_probs + (1 - lam) * student_probs
        mixed_logprobs = torch.log(mixed_probs.clamp_min(eps))

        mask = (no_model_batch != -100).float()
        nonfinite_mask = (~torch.isfinite(logits)) | (~torch.isfinite(teacher_logits))

        prod_probs = torch.masked_fill(teacher_probs * mixed_logprobs, nonfinite_mask, 0)
        x = torch.sum(prod_probs, dim=-1).view(-1)
        denom = torch.sum(mask.view(-1), dim=0).clamp_min(1)
        distil_loss = -torch.sum(x * mask.view(-1), dim=0) / denom
        return distil_loss

    @staticmethod
    def skewed_reverse_kl(logits, teacher_logits, no_model_batch, lam=0.1):
        if logits.size(-1) != teacher_logits.size(-1):
            common_vocab_size = min(logits.size(-1), teacher_logits.size(-1))
            logits = logits[..., :common_vocab_size]
            teacher_logits = teacher_logits[..., :common_vocab_size]
        eps = torch.finfo(torch.float32).eps
        sanitized_logits = torch.where(torch.isfinite(logits), logits, torch.full_like(logits, -1e9))
        sanitized_teacher_logits = torch.where(torch.isfinite(teacher_logits), teacher_logits, torch.full_like(teacher_logits, -1e9))

        teacher_probs = F.softmax(sanitized_teacher_logits, dim=-1, dtype=torch.float32)
        student_probs = F.softmax(sanitized_logits, dim=-1, dtype=torch.float32)
        mixed_probs = (1 - lam) * teacher_probs + lam * student_probs

        student_logprobs = F.log_softmax(sanitized_logits, dim=-1, dtype=torch.float32)
        mixed_logprobs = torch.log(mixed_probs.clamp_min(eps))

        mask = (no_model_batch != -100).float()
        nonfinite_mask = (~torch.isfinite(logits)) | (~torch.isfinite(teacher_logits))

        prod_probs = torch.masked_fill(student_probs * (mixed_logprobs - student_logprobs), nonfinite_mask, 0)
        x = torch.sum(prod_probs, dim=-1).view(-1)
        denom = torch.sum(mask.view(-1), dim=0).clamp_min(1)
        distil_loss = -torch.sum(x * mask.view(-1), dim=0) / denom
        return distil_loss

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

    def _apply_phi(self, x: torch.Tensor, alpha: float, phi_type: str, use_reg: bool = True) -> torch.Tensor:
        if use_reg:
            return self._phi(x, phi_type)
        # Without reg: return linear phi(x) = x, equivalent to pure SFT
        return x

    @staticmethod
    def _safe_metric_value(value: torch.Tensor) -> float:
        safe_value = torch.nan_to_num(value.detach(), nan=0.0, posinf=0.0, neginf=0.0)
        return safe_value.item()

    def _compute_kd_loss(
        self, 
        model, 
        inputs, 
        return_outputs: bool = False,
        num_items_in_batch: Optional[int] = None
    ):
        """
        """
        device = self.accelerator.device
        
        gamma = float(getattr(self.args, "gamma", 0.9999))
        alpha = float(getattr(self.args, "alpha", 0.05))
        phi_type = getattr(self.args, "phi_type", "chi2")
        if phi_type == "chi2_curriculum":
            phi_type = "chi2"
        gamma_tensor = torch.tensor(gamma, dtype=torch.float32, device=device)
        
        input_ids = inputs["input_ids"]  # [batch_size, seq_len]
        attention_mask = inputs.get("attention_mask")  # [batch_size, seq_len]
     
        if attention_mask is None:
            pad_token_id = self.processing_class.pad_token_id if self.processing_class.pad_token_id is not None else 0
            attention_mask = (input_ids != pad_token_id).to(torch.long)
        
        completion_mask = (inputs["labels"] != -100) & (attention_mask.bool())   # [batch_size, seq_len], 0=prompt, 1=completion
   
        
        # ===== normalizer =====
        teacher_token_local = completion_mask.sum().to(torch.float32)
        teacher_token_sum = self.accelerator.reduce(teacher_token_local, reduction="sum")
        normalizer = teacher_token_sum / self.accelerator.num_processes
        

        if normalizer <= 0:
            zero_loss = torch.tensor(0.0, device=device, requires_grad=True)
            if return_outputs:

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                )
                return (zero_loss, outputs)
            else:
                return zero_loss
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )
        
        logits = outputs.logits.float()  # [batch_size, seq_len, vocab_size]
        logits_shift = logits[:, :-1, :]
        
        labels_shift = input_ids[:, 1:]
        completion_mask_shift = completion_mask[:, 1:]  # completion mask [batch_size, seq_len-1]
        
        valid_mask = completion_mask_shift.bool()
        
        # ===== V(s_i) = log sum exp(logits) =====
        V_si_data = torch.logsumexp(logits_shift, dim=-1)  # [batch_size, seq_len-1]
        V_si_data = torch.where(valid_mask, V_si_data, torch.zeros_like(V_si_data))
        
        # ===== log p(a_i | s_i) =====

        vocab_size = logits_shift.size(-1)
        safe_labels = labels_shift.clamp(min=0, max=vocab_size-1)
        
        l_ai_si = torch.gather(
            logits_shift,
            dim=-1,
            index=safe_labels.unsqueeze(-1)
        ).squeeze(-1)  # [batch_size, seq_len-1]
        
        l_ai_si = torch.where(valid_mask, l_ai_si, torch.zeros_like(l_ai_si))
        V_si_data = torch.where(valid_mask, V_si_data, torch.zeros_like(V_si_data))
        
        # ===== V(s_{i+1}) =====
        V_next_data = torch.zeros_like(V_si_data)
        V_next_data[:, :-1] = V_si_data[:, 1:].clone()
        V_next_data = torch.where(valid_mask, V_next_data, torch.zeros_like(V_next_data))
        
        # ===== gamma^t =====
        step_idx = torch.cumsum(valid_mask.to(torch.int32), dim=1) - 1
        step_idx = torch.where(valid_mask, step_idx, torch.full_like(step_idx, -1))
        gamma_pow = torch.pow(gamma_tensor, step_idx.clamp(min=0).to(torch.float32))
        gamma_pow = torch.where(valid_mask, gamma_pow, torch.zeros_like(gamma_pow))
        
        term1_arg = alpha * (l_ai_si - gamma * V_next_data)
        term1 = (1.0 / alpha) * self._apply_phi(term1_arg, alpha, phi_type)
        term2_data = -(V_si_data - gamma * V_next_data)
        
        local_data_loss_sum = (gamma_pow * (term1 + term2_data) * valid_mask).sum()
        

        last_step = step_idx.max(dim=1, keepdim=True).values
        last_mask = valid_mask & (step_idx == last_step)
        V_last_data = torch.where(last_mask, V_si_data, torch.zeros_like(V_si_data)).sum(dim=1)
        gamma_last_data = torch.where(last_mask, gamma_pow, torch.zeros_like(gamma_pow)).sum(dim=1)
        
        if 1.0 - gamma < 1e-6:

            terminal_obj = torch.tensor(0.0, device=device)
        else:
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

            if has_valid.sum() > 0:
                terminal_obj = terminal_data[has_valid].mean()
            else:
                terminal_obj = torch.tensor(0.0, device=device)
    
        del logits, logits_shift, V_si_data, V_next_data
        del gamma_pow, step_idx, valid_mask
        torch.cuda.empty_cache()
        

        data_obj = local_data_loss_sum / normalizer
        loss = -(data_obj + terminal_obj)
        
        if return_outputs:
            return (loss, outputs)
        else:
            return loss



    # Override training step to add activation offloading context.
    def training_step(self, *args, **kwargs):
        with self.maybe_activation_offload_context:
            return super().training_step(*args, **kwargs)

    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        mode = "train" if self.model.training else "eval"
        metrics = {key: sum(val) / len(val) for key, val in self._metrics[mode].items()}  # average the metrics

        # This method can be called both in training and evaluation. When called in evaluation, the keys in `logs`
        # start with "eval_". We need to add the prefix "eval_" to the keys in `metrics` to match the format.
        if mode == "eval":
            metrics = {f"eval_{key}": val for key, val in metrics.items()}

        logs.update(metrics)
        super().log(logs, start_time)
        self._metrics[mode].clear()

    # Ensure the model card is saved along with the checkpoint
    def _save_checkpoint(self, model, trial):
        if self.args.hub_model_id is None:
            model_name = Path(self.args.output_dir).name
        else:
            model_name = self.args.hub_model_id.split("/")[-1]
        self.create_model_card(model_name=model_name)
        super()._save_checkpoint(model, trial)
