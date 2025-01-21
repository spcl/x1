# Copyright (c) 2025 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# main author: Eric Schreiber
#
# contributions: Afonso Catarino

import json
import random
from typing import Any, Callable

import numpy as np
import torch
from tqdm import tqdm
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast


def set_seed(seed: int = 42) -> None:
    """
    Set seed for reproducibility.

    Args:
        seed (int): Seed to set. Defaults to 42.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def preprocess_mcts_dataset_policy(
    dataset: list[dict],
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    max_length: int = 1024,
) -> list[dict]:
    """
    Preprocess and tokenize the dataset for the policy model.

    Args:
        dataset (list[dict]): Dataset.
        tokenizer (PreTrainedTokenizer | PreTrainedTokenizerFast): Tokenizer to use.
        max_length (int): Maximum length of the input sequences. Defaults to 1024.
    Returns:
        list[dict]: Preprocessed dataset.
    """
    processed_data = []
    for example in dataset:
        # Extract reasoning steps and context
        messages = example["messages"]

        # Tokenize and pad/truncate
        tokenized = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            add_generation_prompt=True,
            return_dict=True,
            return_attention_mask=True,
        )
        input_ids = tokenized["input_ids"][0]
        attention_mask = tokenized["attention_mask"][0]

        # Generate collator mask
        effective_len_input_ids = len(input_ids)
        if len(messages) == 1:
            len_until_last_step = effective_len_input_ids
            len_last_step = 0
        else:
            input_id_len_ignore_max = len(
                tokenizer.apply_chat_template(messages, tokenize=True)
            )
            if input_id_len_ignore_max > max_length:
                print(
                    "Input id is longer than max length and was truncated. This should not happen. INCREASING MAX LENGTH ADVISED."
                )
            len_until_last_step = len(
                tokenizer.apply_chat_template(messages[:-1], tokenize=True)
            )
            len_last_step = input_id_len_ignore_max - len_until_last_step
            len_last_step = min(len_last_step, max_length)
            if input_id_len_ignore_max > max_length:
                len_until_last_step = max_length - len_last_step

        collator_mask = torch.cat(
            [torch.zeros(len_until_last_step), torch.ones(len_last_step)]
        )

        # Append processed example
        processed_data.append(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "collator_mask": collator_mask,
                "GAE": example["GAE"],
            }
        )

    return processed_data


def preprocess_mcts_dataset_value(
    dataset: list[dict],
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    max_length: int = 1024,
    loss_padding_value: int = -100,
) -> list[dict]:
    """
    Preprocess and tokenize the dataset for the value model.

    Args:
        dataset (list[dict]): Dataset.
        tokenizer (PreTrainedTokenizer | PreTrainedTokenizerFast): Tokenizer to use.
        max_length (int): Maximum length of the input sequences. Defaults to 1024.
        loss_padding_value (int): Value to use for padding tokens in the loss. Defaults to -100.
    Returns:
        list[dict]: Preprocessed dataset.
    """
    processed_data = []
    for example in dataset:
        # Extract reasoning steps and context
        messages = example["messages"]

        # Tokenize and pad/truncate
        tokenized = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            add_generation_prompt=False,
            return_dict=True,
            return_attention_mask=True,
        )
        input_ids = tokenized["input_ids"][0]
        attention_mask = tokenized["attention_mask"][0]

        # Generate collator mask
        effective_len_input_ids = len(input_ids)
        if len(messages) == 1:
            len_until_last_step = effective_len_input_ids
            len_last_step = 0
        else:
            input_id_len_ignore_max = len(
                tokenizer.apply_chat_template(messages, tokenize=True)
            )
            if input_id_len_ignore_max > max_length:
                print(
                    "Input id is longer than max length and was truncated. This should not happen. INCREASING MAX LENGTH ADVISED."
                )
            len_until_last_step = len(
                tokenizer.apply_chat_template(messages[:-1], tokenize=True)
            )
            len_last_step = input_id_len_ignore_max - len_until_last_step
            len_last_step = min(len_last_step, max_length)
            if input_id_len_ignore_max > max_length:
                len_until_last_step = max_length - len_last_step

        collator_mask = torch.cat(
            [torch.zeros(len_until_last_step), torch.ones(len_last_step)]
        )

        # Create labels tensor with the masking value for the padding tokens
        labels = example["value_target"] * collator_mask + loss_padding_value * (
            1 - collator_mask
        )

        # Append processed example
        processed_data.append(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "collator_mask": collator_mask,
                "labels": labels,
            }
        )

    return processed_data


def get_consecutive_sample_count_with_max_constraint(
    tasks: list[dict], max_batch_size: int
) -> int:
    """
    Get the number of consecutive samples from the start of the list such that
    the count of these samples multiplied by the maximum sample size among them
    is less than or equal to max_batch_size.

    Args:
        tasks (list[dict]): List of tasks with the number of samples in each task.
        max_batch_size (int): Maximum allowable batch size.

    Returns:
        int: Count of consecutive samples that can be included.
    """
    current_max = 0
    count = 0

    for i, task in enumerate(tasks):
        current_max = max(current_max, task[0].get("num_samples", 1))
        if (count + 1) * current_max > max_batch_size:
            break
        count += 1

    return count
