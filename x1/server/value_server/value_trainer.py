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
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
    Trainer,
)

from x1.server import set_seed


class ValueMSETrainer(Trainer):
    """
    Class that implements mean squared error (MSE) training for value models.
    """

    def __init__(
        self,
        value_model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
        device_value_model: str = "cuda:0",
        loss_padding_value: int = -100,
        summary_writer: SummaryWriter = None,
        global_step: int = 0,
    ) -> None:
        """
        Initialize the ValueMSETrainer.

        Args:
            value_model (PreTrainedModel): Value model to train.
            tokenizer (PreTrainedTokenizer | PreTrainedTokenizerFast): Tokenizer to use.
            device_value_model (str): Device to use for the value model. Defaults to cuda:0.
            loss_padding_value (int): Value of the padding token for the loss. Defaults to -100.
            summary_writer (SummaryWriter): Summary writer to use. Defaults to None.
            global_step (int): Global step to start from. Defaults to 0.
        """
        # Value model
        self.value_model = value_model
        self.tokenizer = tokenizer

        self.metrics = {
            "value_loss": [],
        }
        self.summary_writer = summary_writer
        self.global_step = global_step

        self.device_value_model = device_value_model

        self.loss_padding_value = loss_padding_value

    def masked_mean(
        self, tensor: torch.Tensor, mask: torch.Tensor, dim: int | None = None
    ) -> torch.Tensor:
        """
        Compute the mean of a tensor, while ignoring masked values.

        Args:
            tensor (torch.Tensor): Tensor to compute the mean of.
            mask (torch.Tensor): Mask to use.
            dim (int | None): Dimension to compute the mean over. Defaults to None.
        Returns:
            torch.Tensor: Mean of the tensor.
        """
        if dim is not None:
            return (tensor * mask).sum(axis=dim) / (mask.sum(axis=dim) + 1e-8)
        else:
            return (tensor * mask).sum() / (mask.sum() + 1e-8)

    def own_collate_fn_torch(self, batch: list[dict]) -> dict:
        """
        Batch and pad sequences dynamically.

        Args:
            batch (list[dict]): List of samples.
        Returns:
            dict: Dictionary containing the batched and padded sequences.
        """
        max_len = max(len(item["input_ids"]) for item in batch)

        # Pad sequences to `max_len`
        def pad_sequence_tensor(sequence: torch.Tensor, pad_value: int):
            return torch.cat(
                [sequence, torch.tensor([pad_value] * (max_len - len(sequence)))]
            )

        print(f"Keys in batch: {batch[0].keys()}", flush=True)

        # Do the same but with input_ids, attention_mask and collator_mask as tensors
        input_ids = torch.stack(
            [
                pad_sequence_tensor(item["input_ids"], self.tokenizer.pad_token_id)
                for item in batch
            ]
        )
        attention_mask = torch.stack(
            [pad_sequence_tensor(item["attention_mask"], 0) for item in batch]
        )
        collator_mask = torch.stack(
            [pad_sequence_tensor(item["collator_mask"], 0) for item in batch]
        )
        labels = torch.stack(
            [
                pad_sequence_tensor(item["labels"], self.loss_padding_value)
                for item in batch
            ]
        )

        # Convert input_id tensor to int tensor
        input_ids = input_ids.to(torch.int64)
        attention_mask = attention_mask.to(torch.int64)

        return {
            "input_ids": input_ids,  # [batch_size, sequence]
            "attention_mask": attention_mask,  # [batch_size, sequence]
            "collator_mask": collator_mask,  # [batch_size, sequence]
            "labels": labels,  # [batch_size, sequence]
        }

    def train(
        self,
        dataset: Dataset,
        num_iterations: int,
        train_batch_size: int = 2,
        lr: float = 1e-5,
    ) -> int:
        """
        Train the value model on the given dataset.

        Args:
            dataset (Dataset): Dataset to train on.
            num_iterations (int): Number of iterations to train for.
            train_batch_size (int): Batch size to use for training. Defaults to 2.
            lr (float): Learning rate to use for training. Defaults to 1e-5.
        Returns:
            int: Global step after training.
        """

        set_seed(42)
        # Create DataLoader
        train_dataloader = DataLoader(
            dataset,
            batch_size=train_batch_size,
            shuffle=False,  # PER or ER handles sampling! Do not shuffle if you want to avoid duplicates in the batches
            collate_fn=self.own_collate_fn_torch,  # As all inputs are padded to the same length, we can use the default collate_fn
            drop_last=True,  # @trl: needed; otherwise the last batch will be of ragged shape
        )

        value_optimizer = torch.optim.Adam(self.value_model.parameters(), lr=lr)

        with tqdm(
            total=num_iterations, desc="Epoch Progress", dynamic_ncols=True, leave=True
        ) as epoch_progress:
            for epoch in range(num_iterations):
                self.value_model.train()

                value_losses = []

                for step, inputs in enumerate(train_dataloader):
                    # Zero gradients for policy
                    # Policy loss
                    value_optimizer.zero_grad()
                    outputs = self.value_model(
                        input_ids=inputs["input_ids"].to(self.device_value_model),
                        attention_mask=inputs["attention_mask"].to(
                            self.device_value_model
                        ),
                        labels=inputs["labels"].to(self.device_value_model),
                    )
                    outputs.loss.to(dtype=torch.bfloat16).backward()
                    value_optimizer.step()
                    value_losses.append(outputs.loss.item())

                epoch_value_loss = sum(value_losses) / len(value_losses)
                epoch_progress.set_postfix(
                    {
                        "epoch_value_loss": epoch_value_loss,
                    },
                    refresh=True,
                )
                if self.summary_writer is not None:
                    self.summary_writer.add_scalar(
                        "epoch_value_loss", epoch_value_loss, self.global_step
                    )
                self.global_step += 1
                epoch_progress.update(1)

        return self.global_step
