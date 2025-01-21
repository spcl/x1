# Copyright (c) 2025 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# main author: Eric Schreiber
#
# contributions:
# Afonso Catarino
# Robert Gerstenberger

from abc import ABC, abstractmethod
from peft import LoraConfig
from transformers import BatchEncoding
from typing import List, Dict


class AbstractPolicyModel(ABC):
    """
    Abstract base class that defines the interface for accessing hosted policy models.
    """

    @abstractmethod
    def __init__(
        self,
        base_model_path: str,
        LoRa_model_path: str,
        save_path: str,
        lora_training: bool,
        lora_config: LoraConfig,
        device: str,
    ) -> None:
        """
        Initialize the policy model.

        Args:
            base_model_path (str): Path to the base model.
            LoRa_model_path (str): Path to the LoRa model.
            save_path (str): Path to store the output.
            lora_training (bool): Flag to indicate whether LoRa training is enabled.
            lora_config (LoraConfig): Configuration for LoRa training.
            device (str): Device to run the model on
        """
        pass

    @abstractmethod
    def forward(
        self,
        inputs: List[int] | Dict[str, List[int]] | BatchEncoding,
        max_output_length: int,
        num_samples: int,
        temperature: float,
        stop_strings: list[str],
        decoding_strategy: str,
        skip_special_tokens: bool,
    ) -> tuple[list[str], int, list[str]]:
        """
        Simple forward pass through the model.

        Args:
            inputs (List[int] | Dict[str, List[int]] | BatchEncoding): Input data for the model.
            max_output_length (int): Maximum length of the generated output.
            num_samples (int): Number of samples to generate.
            temperature (float): Temperature for sampling.
            stop_strings (list[str]): List of stop strings to use.
            decoding_strategy (str): Decoding strategy to use.
            skip_special_tokens (bool): Whether to skip special tokens in the output.
        Returns:
            tuple[list[str], int, list[str]]: Tuple of the generated texts, logits, and the stop
                                              strings found in the generated texts.
        """

    @abstractmethod
    def forward_chat(
        self,
        input_questions: list[str],
        input_contexts: list[list[str]],
        max_input_length: int,
        max_output_length: int,
        num_samples: int,
        temperature: float,
        decoding_strategy: str,
        stop_strings: list[str],
    ) -> tuple[list[str], int, list[dict], list[str], list[bool]]:
        """
        Get individual reasoning steps from the model for batched questions.

        Args:
            input_questions (list[str]): Input questions for the model.
            input_contexts (list[list[str]]): Previously generated context / steps.
            max_output_length (int): Maximum length of the input.
            max_output_length (int): Maximum length of the generated output.
            num_samples (int): Number of samples to generate.
            temperature (float): Temperature for sampling.
            decoding_strategy (str): Decoding strategy to use.
            stop_strings (list[str]): List of stop strings to use.
        Returns:
            tuple[list[str], int, list[dict],list[str], list[bool]]: Tuple of the generated texts,
                logits, the prompts, the stop strings found in the generated texts, and whether
                steps are terminal.
        """

    @abstractmethod
    def forward_prompt(
        self,
        input_prompt: list[str],
        max_input_length: int,
        max_output_length: int,
        num_samples: int,
        output_logits: bool,
        temperature: float,
        decoding_strategy: str,
        stop_strings: list[str],
    ) -> tuple[list[str], int, list[str]]:
        """
        Tokenize and forward a prompt through the model.

        Args:
            input_prompt (list[str]): Input prompt.
            max_input_length (int): Maximum length of the input.
            max_output_length (int): Maximum length of the generated output.
            num_samples (int): Number of samples to generate.
            output_logits (bool): Whether to output logits.
            temperature (float): Temperature for sampling.
            decoding_strategy (str): Decoding strategy to use.
            stop_strings (list[str]): List of stop strings to use.
        Returns:
            tuple[list[str], int, list[str]]: Tuple of the generated texts, logits if output_logits
                is True, and the stop strings found in the generated texts.
        """

    @abstractmethod
    def get_reasoning_steps(
        self,
        input_questions: list[str],
        input_contexts: list[list[str]],
        max_input_length: int,
        max_output_length: int,
        num_samples: int,
        output_logits: bool,
        temperature: float,
        decoding_strategy: str,
    ) -> tuple[list[str], int, list[dict], list[str], list[bool]]:
        """
        Get reasoning steps from the model without the delimiter "assistant" for batched questions.

        Args:
            input_questions (list[str]): Input question for the model.
            input_contexts (list[list[str]]): Previously generated context / steps.
            max_output_length (int): Maximum length of the input.
            max_output_length (int): Maximum length of the generated output.
            num_samples (int): Number of samples to generate.
            output_logits (bool): Whether to output logits.
            temperature (float): Temperature for sampling.
            decoding_strategy (str): Decoding strategy to use.
        Returns:
            tuple[list[str], int, list[dict],list[str], list[bool]]: Tuple of the generated texts,
                logits if output_logits is True, the prompts, the stop strings found in the
                generated texts, and whether steps are terminal.
        """

    @abstractmethod
    def merge_lora_layers(self) -> None:
        """
        Merge fine-tuned LoRa layers into the base model.
        """

    @abstractmethod
    def ppo_train_on_dataset(
        self,
        samples: list[dict],
        epochs: int,
        learning_rate: float,
        max_length: int,
    ) -> None:
        """
        PPO training on samples.

        Args:
            samples (list[dict]): Samples for fine-tuning.
            epochs (int): Number of epochs to train.
            learning_rate (float): Learning rate for the optimizer.
            max_length (int): Maximum length of the tokenized input sequences.
        """

    @abstractmethod
    def repredict_last_token(self, outputs: dict, stop_token_ids: list[int]) -> dict:
        """
        Efficiently re-predict the last token in the sequence to find the actual stop token.

        Args:
            outputs (dict): Dictionary containing the model outputs.
            stop_token_ids (list[int]): IDs of stop tokens.

        Returns:
            dict: Outputs dictionary with sequences modified to include the most probable stop
                  tokens.
        """

    @abstractmethod
    def simulate_to_end(
        self,
        input_questions: list[str],
        input_contexts: list[list[str]],
        max_input_length: int,
        max_output_length: int,
        num_samples: int,
        output_logits: bool,
        temperature: float,
        decoding_strategy: str,
    ) -> tuple[list[str], int, list[dict], list[str], list[bool]]:
        """
        Generate responses till the end of the conversation.

        Args:
            input_questions (list[str]): Input question for the model.
            input_contexts (list[list[str]]): Previously generated context / steps.
            max_output_length (int): Maximum length of the input.
            max_output_length (int): Maximum length of the generated output.
            num_samples (int): Number of samples to generate.
            output_logits (bool): Whether to output logits.
            temperature (float): Temperature for sampling.
            decoding_strategy (str): Decoding strategy to use.
        Returns:
            tuple[list[str], int, list[dict],list[str], list[bool]]: Tuple of the generated texts,
                logits if output_logits is True, the prompts, the stop strings found in the
                generated texts, and whether steps are terminal.
        """

    @abstractmethod
    def store_model(self, output_path: str) -> None:
        """
        Store model in a directory.

        Args:
            output_path (str): Path to the output directory.
        """
        pass
