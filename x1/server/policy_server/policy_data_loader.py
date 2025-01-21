# Copyright (c) 2025 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# main author: Eric Schreiber
#
# contributions: Afonso Catarino

from typing import Union

from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast


class SFTPolicyDataset(Dataset):
    """
    Implements the Dataset class for the policy model.
    """

    def __init__(
        self,
        data: list[dict],
        tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
        max_length: int = 512,
        name_of_query: str = "query",
        name_of_context: str = "context",
        name_of_label: str = "label",
    ) -> None:
        """
        Initialize the dataset.

        Args:
            data (list[dict]): Dataset.
            tokenizer (PreTrainedTokenizer | PreTrainedTokenizerFast): Tokenizer to use.
            max_length (int): Maximum length of the input. Defaults to 512.
            name_of_query (str): Name of the query field. Defaults to 'query'.
            name_of_context (str): Name of the context field. Defaults to 'context'.
            name_of_label (str): Name of the label field. Defaults to 'label'.
        """

        self.data = data
        self.tokenizer = tokenizer
        # Modify the template to fit the dataset
        template = self.tokenizer.chat_template
        # Add <<Past Reasoning>> to the template
        template = tokenizer.chat_template
        template.replace(
            "{%- if not (message.role == 'ipython' or message.role == 'tool' or 'tool_calls' in message) %}\n",
            "{%- if message.role == 'past_reasoning' %}\n    {{- '<|start_header_id|>' + past reasoning + '<|end_header_id|>\\n\\n'+ message['content'] | trim + '<|eot_id|>' }}\n    {%- elif not (message.role == 'ipython' or message.role == 'tool' or 'tool_calls' in message) %}\n",
        )
        tokenizer.chat_template = template

        self.max_length = max_length
        # Adjust per dataset
        self.name_of_query = name_of_query
        self.name_of_context = name_of_context
        self.name_of_label = name_of_label

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
        # Get the query, past reasoning steps, and next reasoning step
        query = self.data[idx][self.name_of_query]
        past_steps = self.data[idx][self.name_of_context]
        next_step = self.data[idx][self.name_of_label]  # "label"

        tokenized_input = SFTPolicyDataset.apply_chat_template(
            self.tokenizer, query, past_steps, tokenize=True
        )

        tokenized_labels = self.tokenizer(
            next_step,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
            padding="max_length",
        )

        # Set up the labels for language modeling
        labels = tokenized_labels["input_ids"].squeeze(0)
        # We want to ignore the padding tokens in the loss calculation
        labels[labels == self.tokenizer.pad_token_id] = (
            -100
        )  # -100 is the default such that any token with that value is ignored

        return {
            "input_ids": tokenized_input["input_ids"].squeeze(0),
            "attention_mask": tokenized_input["attention_mask"].squeeze(0),
            "labels": labels,
        }

    @staticmethod
    def apply_chat_template(
        tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
        questions: list[str],
        contexts: list[str],
        tokenize: bool = False,
        max_length: int = 512,
    ) -> list[Union[list[int], dict]]:
        """
        Apply the chat template to the questions and contexts.

        Args:
            tokenizer (PreTrainedTokenizer | PreTrainedTokenizerFast): Tokenizer to use.
            questions (list[str]): List of questions.
            contexts (list[str]): List of reasoning steps.
            tokenize (bool): Whether to tokenize the output. Defaults to False.
            max_length (int): Maximum length for padding / truncation. Defaults to 512.
        Returns:
            list[Union[list[int], dict]]: List of chats following the chat template.
        """
        chats = [
            tokenizer.apply_chat_template(
                [
                    {"role": "user", "content": question},
                    {"role": "intermediate_step", "content": context},
                ],
                tokenize=tokenize,
                return_tensors="pt",
                padding="longest",
                truncation=True,
                max_length=max_length,
                add_generation_prompt=False,
            )
            for question, context in zip(questions, contexts)
        ]

        # Remove the last <|eot_id|> if there was no previous context
        for i, chat in enumerate(chats):
            chats[i] = chats[i].replace(
                "assistant<|end_header_id|>\n\n<|eois_id|>",
                "assistant<|end_header_id|>\n\n",
            )

        return chats
