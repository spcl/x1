# Copyright (c) 2025 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# main authors:
# Eric Schreiber
# Julia Barth
#
# contributions: Afonso Catarino

import gc
import os
from typing import Any

import pandas as pd
import torch
import torch.nn as nn
from flask import current_app
from peft import LoraConfig, PeftModelForCausalLM, get_peft_model
from torch.utils.tensorboard import SummaryWriter
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
)
from transformers.modeling_outputs import SequenceClassifierOutput

from datasets import Dataset
from x1.server import preprocess_mcts_dataset_value, set_seed
from x1.server.value_server.value_trainer import ValueMSETrainer
from x1.server.value_server.abstract_value_model import AbstractValueModel


class LinearBlock(nn.Module):
    """
    Last layers of ExtendedValueModel.

    This class inherits from PyTorch's base 'torch.nn.Module' class.
    """

    def __init__(self, in_features: int = 4096, out_features: int = 1) -> None:
        """
        Initialize LinearBlock consisting of 3 linear layers with ReLU activations.

        Args:
            in_features (int): Number of input features. Defaults to 4096.
            out_features (int): Number of output features. Defaults to 1.
        """
        super(LinearBlock, self).__init__()
        self.linear1 = nn.Linear(in_features, 1024).to(dtype=torch.bfloat16)
        self.linear2 = nn.Linear(1024, 256).to(dtype=torch.bfloat16)
        self.linear3 = nn.Linear(256, out_features).to(dtype=torch.bfloat16)
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Simple forward pass through the model.

        Args:
            x (torch.Tensor): Input to LLM.
        Returns:
            torch.Tensor: Output of the last layer.
        """
        x = self.activation(self.linear1(x)).to(dtype=torch.bfloat16)
        x = self.activation(self.linear2(x)).to(dtype=torch.bfloat16)
        return self.linear3(x)

    def save_model(self, path: str) -> None:
        """
        Save model to path.

        Args:
            path (str): Path to save the model to.
        """
        torch.save(self.state_dict(), path)

    def load_model(self, path: str) -> None:
        """
        Load model from path.

        Args:
            path (str): Path to model.
        """
        self.load_state_dict(torch.load(path))


class ExtendedValueModel(torch.nn.Module):
    """
    Class to extend an LLM, by passing its output to a final linear block.

    This class inherits from PyTorch's base 'torch.nn.Module' class.
    """

    def __init__(
        self,
        base_model: PreTrainedModel,
        linear_block: LinearBlock = LinearBlock(),
        loss_padding_value: int = -100,
    ) -> None:
        """
        Initialise extended LLM.

        Args:
            base_model (PreTrainedModel): LLM to extend.
            linear_block (LinearBlock): Final linear block. Defaults to LinearBlock().
            loss_padding_value (int): Padding value for the loss function. Defaults to -100.
        """
        super().__init__()
        self.base_model = base_model
        self.sigmoid = nn.Sigmoid()

        self.loss_padding_value = loss_padding_value

        self.linear_block = linear_block

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
        labels: torch.Tensor = None,
    ) -> SequenceClassifierOutput:
        """
        Simple forward pass through the model.

        Args:
            input_ids (torch.Tensor): Input to LLM.
            attention_mask (torch.Tensor): Mask to apply to attention weights. Defaults to None.
            labels (torch.Tensor): Labels for output. Defaults to None.
        Returns:
            SequenceClassifierOutput: Output of extended model.
        """

        # Get the outputs from the base model (hidden states)
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

        # Pass through the new linear layer
        logits = self.linear_block(outputs.hidden_states[-1][:, :, :])

        # Apply activation function
        logits = 2 * torch.sigmoid(logits) - 1

        # Compute loss if labels are provided
        loss = None
        if labels is not None:
            loss_mask = (labels != self.loss_padding_value).to(dtype=torch.bfloat16)
            loss = nn.MSELoss(reduction="none")(
                logits.squeeze(-1), labels.to(dtype=torch.bfloat16)
            )
            loss = (loss * loss_mask).sum() / (loss_mask.sum() + 1e-8)

        return SequenceClassifierOutput(loss=loss, logits=logits)


class LLMBasedValueModel(AbstractValueModel):
    """
    Class to run a custom value model using a LLM.

    Inherits from the AbstractValueModel class and implements its abstract methods.
    """

    def __init__(
        self,
        base_model_path: str = "meta-llama/Llama-3.1-8B-Instruct",
        LoRa_model_path: str = None,
        save_path: str = "output",
        lora_training: bool = False,
        lora_config: LoraConfig = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        loss_padding_value: int = -100,
    ) -> None:
        """
        Initialize the model.

        Args:
            base_model_path (str): Path to the base model. Defaults to "meta-llama/Llama-3.1-8B-Instruct".
            LoRa_model_path (str): Path to the LoRA model. Defaults to None.
            save_path (str): Path to store the output. Default is "output".
            lora_training (bool): Flag to enable or disable LoRA training. Defaults to False.
            lora_config (LoraConfig): Configuration for LoRa training. Defaults to None.
            device (str): Device to use for training. Defaults to "cuda" if available, otherwise "cpu".
            loss_padding_value (int): Padding value for the loss function. Defaults to -100.
        """
        assert not (
            not lora_training and LoRa_model_path is not None
        ), "LoRa model path is specified but LoRa training is disabled. What do you want to do?"
        assert not (
            not lora_training and lora_config is not None
        ), "LoRa configuration is specified but LoRa training is disabled. What do you want to do?"

        self.device = device
        self.base_model_path = base_model_path
        self.save_path = save_path
        self.lora_training = lora_training
        self.loss_padding_value = loss_padding_value

        self.summary_path = os.path.join(save_path, "training_summary")
        self.writer = SummaryWriter(log_dir=self.summary_path)
        self.global_step = 0

        self.base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_path,
            torch_dtype=torch.bfloat16,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_path, trust_remote_code=True
        )
        self.tokenizer.pad_token = "<|finetune_right_pad_id|>"  # https://discuss.huggingface.co/t/how-to-set-the-pad-token-for-meta-llama-llama-3-models/103418/5

        template = self.tokenizer.chat_template.replace(
            "{%- if not (message.role == 'ipython' or message.role == 'tool' or 'tool_calls' in message) %}",
            "{%- if message.role == 'last_step' %}\n"
            + r"       {{- '<|start_header_id|>' + 'assistant' + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' }}"
            + "\n    {%- elif message.role == 'intermediate_step' %}\n"
            + r"        {{- '<|start_header_id|>' + 'assistant' + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eois_id|>' }}"
            + "\n    {%- elif not (message.role == 'ipython' or message.role == 'tool' or 'tool_calls' in message) %}",
        )
        self.tokenizer.chat_template = template
        # Special token for the end of the intermediate step
        new_tokens = ["<|eois_id|>"]
        # Check if the tokens are already in the vocabulary
        new_tokens = set(new_tokens) - set(self.tokenizer.vocab.keys())
        # Add the tokens to the tokenizer vocabulary
        num_added_tokens = self.tokenizer.add_tokens(
            list(new_tokens), special_tokens=True
        )

        # Set training model: LoRa or base model
        if self.lora_training:
            if LoRa_model_path is not None:
                print("Loading LoRa model from file.", flush=True)
                self.tokenizer = AutoTokenizer.from_pretrained(
                    LoRa_model_path, trust_remote_code=True
                )
                self.base_model.resize_token_embeddings(len(self.tokenizer))
                self.llm = PeftModelForCausalLM.from_pretrained(
                    model_id=LoRa_model_path,
                    model=self.base_model,
                    torch_dtype=torch.bfloat16,
                    is_trainable=True,
                )
            else:
                if lora_config is None:
                    # Default LoRa configuration
                    self.lora_config = LoraConfig(
                        r=8,
                        lora_alpha=16,
                        lora_dropout=0.05,
                    )
                else:
                    self.lora_config = lora_config
                self.llm = get_peft_model(self.base_model, self.lora_config)
            print("LoRA model initialized successfully.", flush=True)
        else:
            self.llm = self.base_model
            print("Model initialized successfully.", flush=True)

        new_output_dim = 1
        hidden_size = self.llm.config.hidden_size

        self.linear_block = LinearBlock(hidden_size, new_output_dim)
        if LoRa_model_path is not None and os.path.exists(
            os.path.join(LoRa_model_path, "linear_block.pth")
        ):
            print("Loading linear block from file.", flush=True)
            self.linear_block.load_model(
                os.path.join(LoRa_model_path, "linear_block.pth")
            )

        self.model = ExtendedValueModel(
            self.llm,
            linear_block=self.linear_block,
            loss_padding_value=loss_padding_value,
        ).to(self.device)

    def store_model(self, output_path: str = None) -> None:
        """
        Store model in a directory.

        Args:
            output_path (str): Path to the output directory. Defaults to None.
        """
        self.model.base_model.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)
        self.model.linear_block.save_model(
            os.path.join(output_path, "linear_block.pth")
        )
        print(f"Model and tokenizer saved to {output_path}", flush=True)

    def forward(self, inputs: Any, position_plus_x: int = -3) -> list[float]:
        """
        Simple forward pass through the model.

        Args:
            inputs (Any): Input data for the model.
            position_plus_x (int): Postion of where to start the evaluation of the input. Defaults to -3.
        Returns:
            list[float]: Generated values.
        """
        self.model.eval()
        last_non_pad_token_idx = (inputs["attention_mask"] == 1).sum(dim=1) - 1
        # Through evaluation we saw that three tokens before the <eot> or <eois> token is the best to extract the value.
        positions_to_extract = torch.max(
            last_non_pad_token_idx + position_plus_x,
            torch.zeros_like(last_non_pad_token_idx),
        )

        outputs = self.model(**inputs).logits

        # Extract the value from the last hidden state
        outputs = outputs[range(outputs.size(0)), positions_to_extract]
        return outputs

    def evaluate_prompt(
        self, prompts: list[str], max_input_length: int = 1024
    ) -> list[float]:
        """
        Evaluate prompts.

        Args:
            prompts (list[str]): List of input prompts to be evaluated by the model.
            max_input_length (int): Maximum length of the input sequences. Defaults to 1024.
        Returns:
            list[float]: Generated values.
        """
        input = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=max_input_length,
            return_attention_mask=True,
        ).to(self.device)
        values = self.forward(input)
        # Convert batch to list
        values = values.to(torch.float32).cpu().detach().numpy().flatten().tolist()
        return values

    def evaluate_reasoning_steps(
        self, messages: list[list[dict]], max_input_length=1024
    ) -> list[float]:
        """
        Evaluate individual reasoning steps.

        Args:
            messages (list[list[dict]]): List of reasoning steps to be evaluated by the model.
            max_input_length (int): Maximum length of the input sequences. Defaults to 1024.
        Returns:
            list[float]: Generated values.
        """
        input = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=max_input_length,
            add_generation_prompt=False,
            return_dict=True,
            return_attention_mask=True,
        ).to(self.device)

        values = self.forward(input)
        # Convert batch to list
        values = values.to(torch.float32).cpu().detach().numpy().flatten().tolist()
        return values

    def mse_train_on_dataset(
        self,
        samples: list[dict],
        epochs: int = 2,
        learning_rate: float = 1e-5,
        max_length: int = 1024,
    ) -> None:
        """
        MSE training on samples.

        Args:
            samples (list[dict]): Samples for fine-tuning.
            epochs (int): Number of epochs to train. Defaults to 2.
            learning_rate (float): Learning rate for the optimizer. Defaults to 1e-5.
            max_length (int): Maximum length of the tokenized input sequences. Defaults to 1024.
        """
        self.model.train()
        set_seed()
        samples = pd.DataFrame(samples)
        samples = Dataset.from_pandas(samples)
        processed_data = preprocess_mcts_dataset_value(
            dataset=samples, tokenizer=self.tokenizer, max_length=max_length
        )
        mse_trainer = ValueMSETrainer(
            value_model=self.model,
            tokenizer=self.tokenizer,
            device_value_model=self.device,
            summary_writer=self.writer,
            global_step=self.global_step,
        )
        print(
            f"Value Model GPU summary before training: {torch.cuda.memory_summary(device=self.device)}",
            flush=True,
        )
        self.global_step = mse_trainer.train(
            processed_data,
            num_iterations=epochs,
            train_batch_size=current_app.config["BATCH_SIZE_TRAIN"],
            lr=learning_rate,
        )
        self.writer.flush()

        del processed_data
        del mse_trainer
        gc.collect()
        torch.cuda.empty_cache()
        print(
            f"Value Model GPU summary after training: {torch.cuda.memory_summary(device=self.device)}",
            flush=True,
        )
        self.save_model(
            save_path=os.path.join(self.save_path, "iteration-" + str(self.global_step))
        )
