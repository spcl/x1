# Copyright (c) 2025 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# main author: Eric Schreiber
#
# contributions: Afonso Catarino

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast


class RewardDataset(Dataset):
    """
    Implements the Dataset class for the reward model.
    """

    def __init__(
        self,
        data: list[dict],
        tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
        max_length: int = 1024,
    ) -> None:
        """
        Initialize the dataset.

        Args:
            data (list[dict]): Dataset.
            tokenizer (PreTrainedTokenizer | PreTrainedTokenizerFast): Tokenizer to use.
            max_length (int): Maximum length of the input.
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        """
        Return the size of the dataset.

        Returns:
            int: Size of the dataset.
        """
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        """
        Return the item at the given index.

        Args:
            idx (int): Index of the item to retrieve.
        Returns:
            dict: Item at the given index.
        """
        conversation = self.data[idx]["prompt"]
        label = self.data[idx]["label"]

        # Convert the conversation into a single text string
        text = self.convert_conversation_to_text(conversation)

        # Tokenize the text
        tokenized_input = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
            padding="max_length",
        )

        label_tensor = torch.tensor([label], dtype=torch.float16)

        return {
            "input_ids": tokenized_input["input_ids"].squeeze(0),
            "attention_mask": tokenized_input["attention_mask"].squeeze(0),
            "labels": label_tensor,
        }

    @staticmethod
    def convert_conversation_to_text(conversation: list[dict]) -> str:
        """
        Convert a user - assistant conversation into a single text string.

        Args:
            conversation (list[dict]): User - assistant conversation.
        Returns:
            str: Text string.
        """
        text = ""
        for turn in conversation:
            if turn["role"] == "user":
                text += f"<s>[INST] {turn['content']} [/INST] "
            elif turn["role"] == "assistant":
                text += f"{turn['content']} </s>"
        return text


def create_data_loader(
    data: list[dict],
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    batch_size: int = 32,
    max_length: int = 512,
) -> DataLoader:
    """
    Create DataLoader for the given data.

    Args:
        data (list[dict]): Data to load.
        tokenizer (PreTrainedTokenizer | PreTrainedTokenizerFast): Tokenizer to use.
        batch_size (int): Batch size. Defaults to 32.
        max_length (int): Maximum length of the input. Defaults to 512.
    Returns:
        DataLoader: DataLoader for the given data.
    """
    dataset = RewardDataset(data, tokenizer, max_length)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)
