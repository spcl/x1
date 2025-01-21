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
from typing import Any


class AbstractValueModel(ABC):
    """
    Abstract base class that defines the interface for accessing hosted value models.
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
        loss_padding_value: int,
    ) -> None:
        """
        Initialize the model.

        Args:
            base_model_path (str): Path to the base model.
            LoRa_model_path (str): Path to the LoRA model.
            save_path (str): Path to store the output.
            lora_training (bool): Flag to enable or disable LoRA training.
            lora_config (LoraConfig): Configuration for LoRa training.
            device (str): Device to use for training.
            loss_padding_value (int): Padding value for the loss function.
        """

    @abstractmethod
    def forward(self, inputs: Any, position_plus_x: int) -> list[float]:
        """
        Simple forward pass through the model.

        Args:
            inputs (Any): Input data for the model.
            position_plus_x (int): Postion of where to start the evaluation of the input.
        Returns:
            list[float]: Generated values.
        """

    @abstractmethod
    def evaluate_prompt(self, prompts: list[str], max_input_length: int) -> list[float]:
        """
        Evaluate prompts.

        Args:
            prompts (list[str]): List of input prompts to be evaluated by the model.
            max_input_length (int): Maximum length of the input sequences.
        Returns:
            list[float]: Generated values.
        """

    @abstractmethod
    def evaluate_reasoning_steps(
        self, messages: list[list[dict]], max_input_length
    ) -> list[float]:
        """
        Evaluate individual reasoning steps.

        Args:
            messages (list[list[dict]]): List of reasoning steps to be evaluated by the model.
            max_input_length (int): Maximum length of the input sequences.
        Returns:
            list[float]: Generated values.
        """

    @abstractmethod
    def mse_train_on_dataset(
        self,
        samples: list[dict],
        epochs: int,
        learning_rate: float,
        max_length: int,
    ) -> None:
        """
        MSE training on samples.

        Args:
            samples (list[dict]): Samples for fine-tuning.
            epochs (int): Number of epochs to train.
            learning_rate (float): Learning rate for the optimizer.
            max_length (int): Maximum length of the tokenized input sequences.
        """

    @abstractmethod
    def store_model(self, output_path: str) -> None:
        """
        Store model in a directory.

        Args:
            output_path (str): Path to the output directory.
        """
        pass
