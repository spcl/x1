# Copyright (c) 2025 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# main authors:
# Julia Barth
# Eric Schreiber
#
# contributions: Afonso Catarino

from copy import deepcopy

import torch
import torch.nn.functional as F
from peft import PeftModel
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


class PolicyPPOTrainer(Trainer):
    def __init__(
        self,
        policy_model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
        epsilon: float = 0.2,
        lambda_kl: float = 0.01,
        lambda_e: float = 0.01,
        value_clip: float = 0.2,
        logit_temp: float = 0.9,
        device_policy_model: str = "cuda:0",
        device_old_policy_model: str = "cuda:1",
        summary_writer: SummaryWriter = None,
        global_step: int = 0,
    ) -> None:
        """
        Initialize the PolicyPPOTrainer.

        Args:
            policy_model (PreTrainedModel): Policy model to train.
            tokenizer (PreTrainedTokenizer | PreTrainedTokenizerFast): Tokenizer to use.
            epsilon (float): PPO clip parameter. Defaults to 0.2.
            lambda_kl (float): Weight factor of the KL divergence between the current model and the
                               model before RL training started. Defaults to 0.01.
            lambda_e (float): Weight factor for the entropy loss of the log probabilities produced
                              by the current model. Defaults to 0.01.
            value_clip (float): Value clip parameter. Defaults to 0.2.
            logit_temp (float): Logit temperature. Defaults to 0.9.
            device_policy_model (str): Device to use for the policy model.
            device_old_policy_model (str): Device to use for the old policy model.
            summary_writer (SummaryWriter): Summary writer to use. Defaults to None.
            global_step (int): Global step to start from. Defaults to 0.
        Raises:
            ValueError: Raises an exception.
        """

        # Policy models
        self.last_epoch_model = deepcopy(policy_model).to(device_old_policy_model)
        self.model = policy_model.to(device_policy_model)
        self.tokenizer = tokenizer
        self.vocab_size = self.model.config.vocab_size

        self.summary_writer = summary_writer
        self.global_step = global_step

        # Note: Our reference model is a disabled LoRa version of our model.
        # Check if model is a PEFT model such that it is trained only with LoRa.
        # If not either init as such or give an error.
        if not isinstance(self.model, PeftModel):
            raise ValueError(
                "Your model is not a PEFT model. Please provide a PEFT model as policy_model."
            )

        self.epsilon = epsilon
        self.lambda_kl = lambda_kl
        self.lambda_e = lambda_e
        self.value_clip = value_clip
        self.logit_temp = logit_temp

        self.metrics = {
            "policy_loss": [],
            "entropy": [],
            "kl_divergence": [],
            "total_policy_loss": [],
        }

        self.device_policy_model = device_policy_model
        self.device_old_policy_model = device_old_policy_model

        self.last_epoch_model.eval()  # Evaluation mode for memory

    def own_collate_fn(self, batch: list[dict]) -> dict:
        """
        Batch and pad sequences dynamically.

        Args:
            batch (list[dict]): List of samples.
        Returns:
            dict: Dictionary with the batched and padded sequences.
        """
        max_len = max(len(item["input_ids"]) for item in batch)

        # Pad sequences to `max_len`
        def pad_sequence_tensor(sequence: torch.Tensor, pad_value: int):
            return torch.cat(
                [sequence, torch.tensor([pad_value] * (max_len - len(sequence)))]
            )

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

        # Convert input_id tensor to int tensor
        input_ids = input_ids.to(torch.int64)
        attention_mask = attention_mask.to(torch.int64)

        return {
            "input_ids": input_ids,  # [batch_size, sequence]
            "attention_mask": attention_mask,  # [batch_size, sequence]
            "collator_mask": collator_mask,  # [batch_size, sequence]
            "GAE": torch.tensor([item["GAE"] for item in batch]),  # [batch_size]
            "value_target": torch.tensor(
                [item["GAE"] for item in batch]
            ),  # [batch_size]
        }

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

    def compute_entropy(
        self, new_policy_log_probs: torch.Tensor, collator_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the entropy loss.

        Args:
            new_policy_log_probs (torch.Tensor): Log probabilities of the new policy.
            collator_mask (torch.Tensor): Mask to use.
        Returns:
            torch.Tensor: Entropy loss.
        """
        # We calculate the entropy not over the whole vocab distribution.
        new_policy_probs = torch.exp(new_policy_log_probs)
        entropy_loss = -(new_policy_probs * new_policy_log_probs).sum(
            dim=-1
        )  # over vocab
        entropy_loss = self.masked_mean(
            entropy_loss, collator_mask, 1
        )  # Sequence-level masking + mean
        entropy_loss = entropy_loss.mean()  # Batch-level mean
        return entropy_loss

    def compute_policy_loss_target(
        self, inputs: dict, disable_ratio: bool = False
    ) -> torch.Tensor:
        """
        Compute the loss, incorporating importance-sampling weights for target tokens.

        Args:
            inputs (dict): Inputs to the model.
            disable_ratio (bool): Whether to disable the ratio calculation.
        Returns:
            torch.Tensor: Total loss.
        """

        # Move inputs to device
        input_ids = inputs["input_ids"].to(
            self.device_policy_model
        )  # Full sequence (context + generated tokens)
        attention_mask = inputs["attention_mask"].to(self.device_policy_model)
        target_ids = inputs["input_ids"][:, 1:].to(
            self.device_policy_model
        )  # Shifted targets for next-token prediction
        collator_mask_full = inputs["collator_mask"].to(
            self.device_policy_model
        )  # Mask for valid steps - outcomes
        collator_mask = collator_mask_full[:, :-1]
        advantage = inputs["GAE"].to(self.device_policy_model)  # Token-level advantages

        epsilon = self.epsilon  # PPO clip parameter

        # 1. Policy loss using PPO
        if disable_ratio:
            # For KL and Entropy
            new_policy_logits = self.model(input_ids, attention_mask).logits
            new_policy_logits /= self.logit_temp + 1e-7
            new_policy_log_probs = F.log_softmax(new_policy_logits, dim=-1)
            selected_log_probs = (
                new_policy_log_probs[:, :-1]
                .gather(2, target_ids.unsqueeze(-1))
                .squeeze(-1)
            )  # [batch, sequence]

            # Ratio is 1
            batch_size, sequence = input_ids.shape
            ratio = torch.ones(
                batch_size, sequence - 1, device=self.device_policy_model
            )  # [batch_size, sequence]

        else:
            # For KL, Ratio and Entropy
            new_policy_logits = self.model(input_ids, attention_mask).logits
            new_policy_logits /= self.logit_temp + 1e-7
            new_policy_log_probs = F.log_softmax(new_policy_logits, dim=-1)
            selected_log_probs = (
                new_policy_log_probs[:, :-1]
                .gather(2, target_ids.unsqueeze(-1))
                .squeeze(-1)
            )
            # For Ratio
            with torch.no_grad():
                old_policy_logits = self.last_epoch_model(
                    input_ids.to(self.device_old_policy_model),
                    attention_mask.to(self.device_old_policy_model),
                ).logits
                old_policy_logits /= self.logit_temp + 1e-7
                old_policy_log_probs = F.log_softmax(old_policy_logits, dim=-1)
                selected_old_log_probs = (
                    old_policy_log_probs[:, :-1]
                    .gather(
                        2, target_ids.to(self.device_old_policy_model).unsqueeze(-1)
                    )
                    .squeeze(-1)
                )

            ratio = (
                selected_log_probs - selected_old_log_probs.to(self.device_policy_model)
            ).exp()  # [batch_size, sequence]

        advantage = advantage.view(-1, 1)  # [batch_size, 1]
        surr1 = ratio * advantage  # Broadcasting for token-level GAE
        surr2 = torch.clamp(ratio, min=1 - epsilon, max=1 + epsilon) * advantage
        pl_loss = -torch.min(surr1, surr2)

        # 1. Policy loss:
        collator_mask_bsv = collator_mask  # [batch_size, sequence]
        policy_loss = self.masked_mean(
            pl_loss, collator_mask_bsv, dim=1
        )  # [batch_size, vocab]
        policy_loss = policy_loss.mean()
        self.metrics["policy_loss"].append(policy_loss.clone().detach().item())

        # 2. Entropy loss
        # We calculate the entropy not over the whole vocab distribution.
        if self.lambda_e != 0:
            entropy_loss = self.compute_entropy(
                new_policy_log_probs, collator_mask_full
            )
            self.metrics["entropy"].append(entropy_loss.clone().detach().item())
        else:
            entropy_loss = 0

        # 3. KL penalty
        if self.lambda_kl != 0:
            with torch.no_grad():
                # Get the starting logits by removing the lora layer
                starting_logits = (
                    self.model.get_base_model()
                    .forward(input_ids, attention_mask)
                    .logits
                )
                starting_logits /= self.logit_temp + 1e-7
                starting_log_probs = F.log_softmax(starting_logits, dim=-1)
                selected_starting_log_probs = (
                    starting_log_probs[:, :-1]
                    .gather(2, target_ids.unsqueeze(-1))
                    .squeeze(-1)
                )

            kl_div = torch.exp(selected_starting_log_probs) * (
                selected_starting_log_probs - selected_log_probs
            )
            kl_div = self.masked_mean(kl_div, collator_mask, 1)  # Sequence-level mean
            kl_div = kl_div.mean()  # Batch-level mean
            self.metrics["kl_divergence"].append(kl_div.clone().detach().item())
        else:
            kl_div = 0

        # Combined loss
        total_loss = (
            policy_loss - self.lambda_e * entropy_loss + self.lambda_kl * kl_div
        )
        self.metrics["total_policy_loss"].append(total_loss.clone().detach().item())

        return total_loss

    def compute_policy_loss(
        self, inputs: dict, disable_ratio: bool = False
    ) -> torch.Tensor:
        """
        Compute the loss.

        Args:
            inputs (dict): Inputs to the model.
            disable_ratio (bool): Whether to disable the ratio calculation. Defaults to False.
        Returns:
            torch.Tensor: Total loss.
        """

        input_ids = inputs["input_ids"].to(
            self.device_policy_model
        )  # [batch_size, sequence]
        attention_mask = inputs["attention_mask"].to(
            self.device_policy_model
        )  # [batch_size, sequence]
        collator_mask = inputs["collator_mask"].to(
            self.device_policy_model
        )  # [batch_size, sequence]
        advantage = inputs["GAE"].to(self.device_policy_model)  # [batch_size]

        epsilon = self.epsilon  # PPO clip parameter

        # 1. Policy loss using PPO
        if disable_ratio:
            # For KL and Entropy
            if self.lambda_kl != 0 or self.lambda_e != 0:
                new_policy_logits = self.model(input_ids, attention_mask).logits
                new_policy_logits /= self.logit_temp + 1e-7
                new_policy_log_probs = F.log_softmax(new_policy_logits, dim=-1)
            # Ratio is 1
            batch_size, sequence = input_ids.shape
            ratio = torch.ones(
                batch_size, sequence, self.vocab_size, device=self.device_policy_model
            )  # [batch_size, sequence, vocab]
        else:
            # For KL, Ratio and Entropy
            new_policy_logits = self.model(input_ids, attention_mask).logits
            new_policy_logits /= self.logit_temp + 1e-7
            new_policy_log_probs = F.log_softmax(new_policy_logits, dim=-1)
            # For Ratio
            with torch.no_grad():
                old_policy_logits = self.last_epoch_model(
                    input_ids.to(self.device_old_policy_model),
                    attention_mask.to(self.device_old_policy_model),
                ).logits
                old_policy_logits /= self.logit_temp + 1e-7
                old_policy_log_probs = F.log_softmax(old_policy_logits, dim=-1)

            ratio = (
                new_policy_log_probs - old_policy_log_probs.to(self.device_policy_model)
            ).exp()  # [batch_size, sequence, vocab]

        advantage = advantage.view(-1, 1, 1)  # [batch_size, 1, 1]
        surr1 = -ratio * advantage  # Broadcasting for token-level GAE
        surr2 = -torch.clamp(ratio, min=1 - epsilon, max=1 + epsilon) * advantage
        pl_loss = torch.max(surr1, surr2)

        # 1. Policy loss:
        collator_mask_bsv = collator_mask.unsqueeze(-1)  # [batch_size, sequence, 1]
        policy_loss = self.masked_mean(
            pl_loss, collator_mask_bsv, dim=1
        )  # [batch_size, vocab]
        policy_loss = policy_loss.mean()
        self.metrics["policy_loss"].append(policy_loss.clone().detach().item())

        # 2. Entropy loss:
        if self.lambda_e != 0:
            entropy_loss = self.compute_entropy(new_policy_log_probs, collator_mask)
            self.metrics["entropy"].append(entropy_loss.clone().detach().item())
        else:
            entropy_loss = 0

        # 3. KL penalty:
        if self.lambda_kl != 0:
            with torch.no_grad():
                # Get the starting logits by removing the LoRa layer
                starting_logits = (
                    self.model.get_base_model()
                    .forward(input_ids, attention_mask)
                    .logits
                )
                starting_logits /= self.logit_temp + 1e-7
                starting_log_probs = F.log_softmax(starting_logits, dim=-1)

            kl_div = torch.exp(starting_log_probs) * (
                starting_log_probs - new_policy_log_probs
            )
            kl_div = kl_div.sum(
                dim=-1
            )  # Sum over vocab dimension: [batch_size, seq_len]
            kl_div = self.masked_mean(kl_div, collator_mask, 1)  # Sequence-level mean
            kl_div = kl_div.mean()  # Batch-level mean
            self.metrics["kl_divergence"].append(kl_div.clone().detach().item())
        else:
            kl_div = 0

        # Combined loss: scalar
        total_loss = (
            policy_loss - self.lambda_e * entropy_loss + self.lambda_kl * kl_div
        )
        self.metrics["total_policy_loss"].append(total_loss.clone().detach().item())

        return total_loss

    def train(
        self,
        dataset: Dataset,
        num_iterations: int,
        train_batch_size: int = 2,
        lr: float = 1e-5,
    ) -> int:
        """
        Train the policy model on the given dataset.

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
            shuffle=False,  # PER or ER handles sampling! Do not shuffle if you want to avoid duplicates in the batches.
            collate_fn=self.own_collate_fn,
            drop_last=True,  # @trl: needed; otherwise the last batch will be of ragged shape
        )

        policy_optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        with tqdm(
            total=num_iterations, desc="Epoch Progress", dynamic_ncols=True, leave=True
        ) as epoch_progress:
            for epoch in range(num_iterations):
                self.last_epoch_model.eval()  # Evaluation mode for memory
                self.model.train()

                policy_losses = []

                for step, inputs in enumerate(train_dataloader):
                    # Zero gradients for policy
                    # Policy loss
                    policy_optimizer.zero_grad()
                    masked_mean_loss = self.compute_policy_loss_target(
                        inputs, disable_ratio=(step == 0)
                    )
                    masked_mean_loss.backward()
                    policy_optimizer.step()
                    policy_losses.append(masked_mean_loss.detach().item())

                epoch_policy_loss = sum(policy_losses) / len(policy_losses)
                epoch_progress.set_postfix(
                    {
                        "epoch_policy_loss": epoch_policy_loss,
                    },
                    refresh=True,
                )

                if self.summary_writer is not None:
                    self.summary_writer.add_scalar(
                        "epoch_policy_loss", epoch_policy_loss, self.global_step
                    )
                self.global_step += 1
                epoch_progress.update(1)

        return self.global_step
