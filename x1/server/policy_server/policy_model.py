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

import numpy as np
import pandas as pd
import torch
from flask import current_app
from peft import LoraConfig, PeftModelForCausalLM, get_peft_model
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoModelForCausalLM, AutoTokenizer, BatchEncoding

from datasets import Dataset
from x1.server import preprocess_mcts_dataset_policy, set_seed
from x1.server.policy_server.policy_trainer import PolicyPPOTrainer
from x1.server.policy_server.abstract_policy_model import AbstractPolicyModel


class PolicyModel(AbstractPolicyModel):
    """
    The PolicyModel class is responsible for hosting and running a policy model and provides
    methods for both training and inference.

    Inherits from the AbstractPolicyModel class and implements its abstract methods.
    """

    def __init__(
        self,
        base_model_path: str = "meta-llama/Llama-3.1-8B-Instruct",
        LoRa_model_path: str = None,
        save_path: str = "output",
        lora_training: bool = False,
        lora_config: LoraConfig = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ) -> None:
        """
        Initialize the policy model.

        Args:
            base_model_path (str): Path to the base model. Defaults to "meta-llama/Llama-3.1-8B-Instruct".
            LoRa_model_path (str): Path to the LoRa model. Defaults to None.
            save_path (str): Path to store the output. Defaults to "output".
            lora_training (bool): Flag to indicate whether LoRa training is enabled. Defaults to False.
            lora_config (LoraConfig): Configuration for LoRa training. Defaults to None.
            device (str): Device to run the model on. Defaults to "cuda" if available, otherwise "cpu".
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

        # Set training model: LoRA or base model
        if self.lora_training:
            if LoRa_model_path is not None:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    LoRa_model_path, trust_remote_code=True
                )
                self.base_model.resize_token_embeddings(len(self.tokenizer))
                self.model = PeftModelForCausalLM.from_pretrained(
                    model_id=LoRa_model_path,
                    model=self.base_model,
                    torch_dtype=torch.bfloat16,
                    device_map="auto",  # 'cuda'
                    is_trainable=True,
                )
            else:
                if lora_config is None:
                    # Default LoRA configuration
                    self.lora_config = LoraConfig(
                        r=8,
                        lora_alpha=16,
                        lora_dropout=0.05,
                    )
                else:
                    self.lora_config = lora_config
                self.model = get_peft_model(self.base_model, self.lora_config)
            print("LoRA model initialized successfully.", flush=True)
        else:
            self.model = self.base_model
            print("Model initialized successfully.", flush=True)

        self.model.to(self.device)

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
        # Get sequences and identify stop token positions (keep everything on GPU)
        sequences = outputs["sequences"]  # Already on the GPU
        stop_token_ids = torch.tensor(
            stop_token_ids, device=self.device
        )  # Move stop tokens to GPU
        mask = torch.isin(sequences, stop_token_ids)  # Mask where stop tokens appear
        # Find the index of the last stop token for each sequence
        indices = (
            torch.arange(sequences.size(1), device=self.device)
            .unsqueeze(0)
            .expand_as(sequences)
        )
        indices = torch.where(mask, indices, torch.tensor(-1, device=self.device))
        largest_stop_token_idx = indices.max(
            dim=1
        ).values  # Row-wise max to get the last stop token index

        # Make a forward pass to get logits for the entire sequence
        outputs_repredict = self.model(sequences)
        logits = (
            outputs_repredict.logits
        )  # [batchsize*num_samples, seq_len, vocab_size]

        # Select logits for stop token candidates at the relevant indices
        valid_rows = (
            largest_stop_token_idx >= 0
        )  # Mask for sequences with valid stop tokens
        rows = torch.arange(sequences.size(0), device=self.device)[
            valid_rows
        ]  # Row indices
        cols = (
            largest_stop_token_idx[valid_rows] - 1
        )  # Column indices (previous token positions)
        stop_logits = logits[rows, cols][:, stop_token_ids]  # Logits for stop_token_ids

        # Find the most probable stop token for each valid sequence
        max_indices = stop_logits.argmax(dim=1)  # Index within stop_token_ids
        chosen_stop_tokens = stop_token_ids[max_indices]  # Convert indices to token IDs

        # Update the sequences in place
        outputs["sequences"][
            rows, largest_stop_token_idx[valid_rows]
        ] = chosen_stop_tokens

        return outputs

    def forward(
        self,
        inputs: list[int] | dict[str, list[int]] | BatchEncoding,
        max_output_length: int = 1152,
        num_samples: int = 1,
        temperature: float = 0.7,
        stop_strings: list[str] = ["<|eot_id|>"],
        decoding_strategy: str = "diverse_beam_search",
        skip_special_tokens: bool = True,
    ) -> tuple[list[str], int, list[str]]:
        """
        Simple forward pass through the model.

        Note: eot_id = 128009, eois_id = 128256

        Args:
            inputs (list[int] | dict[str, list[int]] | BatchEncoding): Input data for the model.
            max_output_length (int): Maximum length of the generated output. Defaults to 1152.
            num_samples (int): Number of samples to generate. Defaults to 1.
            temperature (float): Temperature for sampling. Defaults to 0.7.
            stop_strings (list[str]): List of stop strings to use. Defaults to ["<|eot_id|>"].
            decoding_strategy (str): Decoding strategy to use. Defaults to "diverse_beam_search".
            skip_special_tokens (bool): Whether to skip special tokens in the output. Defaults to True.
        Returns:
            tuple[list[str], int, list[str]]: Tuple of the generated texts, -1, and the stop strings
                                              found in the generated texts.
        Raises:
            ValueError: Raises an exception.
        """
        self.model.eval()
        # Generate the stop token ids from the stop strings
        stop_token_ids = []
        for stop_string in stop_strings:
            tokenised_stop_string = self.tokenizer(stop_string)["input_ids"]
            assert (
                len(tokenised_stop_string) == 2
            ), f"Multi token stop strings were not tested. Be careful. Current stop string: {stop_string}."
            stop_token_ids.append(tokenised_stop_string[1])
        stop_token_ids = np.array(stop_token_ids)

        inputs.to(self.device)

        if num_samples == 1 and decoding_strategy != "temperature":
            print(
                "Warning: num_samples is 1 but generation mode is not temperature. Setting generation mode to temperature.",
                flush=True,
            )
            decoding_strategy = "temperature"

        if decoding_strategy == "temperature" or num_samples == 1:
            outputs = self.model.generate(
                **inputs,
                max_length=max_output_length,
                num_return_sequences=num_samples,
                return_dict_in_generate=True,
                output_logits=False,
                temperature=temperature,
                stop_strings=stop_strings,
                tokenizer=self.tokenizer,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        elif decoding_strategy == "diverse_beam_search":
            assert num_samples > 1, "Diverse beam search requires num_samples > 1."
            outputs = self.model.generate(
                **inputs,
                max_length=max_output_length,
                num_return_sequences=num_samples,
                return_dict_in_generate=True,
                output_logits=False,
                pad_token_id=self.tokenizer.pad_token_id,
                do_sample=False,
                num_beams=num_samples,
                num_beam_groups=num_samples,
                diversity_penalty=1.0,
                eos_token_id=stop_token_ids,
            )
            # Because Diverse Beam Search just adds the first stop token and we need to know which
            # one it actually is, we repredict the last token.

            outputs = self.repredict_last_token(outputs, stop_token_ids)
        else:
            raise ValueError(
                f"Generation mode not supported. Current mode: {decoding_strategy}, supported modes: ['temperature', 'diverse_beam_search']."
            )

        generated_texts = self.tokenizer.batch_decode(
            outputs["sequences"], skip_special_tokens=skip_special_tokens
        )

        print(f"Generated texts: {generated_texts}", flush=True)

        generated_texts_special_tokens = self.tokenizer.batch_decode(
            outputs["sequences"], skip_special_tokens=False
        )

        # Check if eois or eot comes first in the string when looking from the end
        def find_stop_string(text):
            idxes = [text.rfind(stop_string) for stop_string in stop_strings]
            # Find the position of the maximum index
            idx = max(idxes)
            return text[idx : idx + len(stop_strings[idxes.index(idx)])]

        stop_strings_found = [
            find_stop_string(text) for text in generated_texts_special_tokens
        ]
        return generated_texts, -1, stop_strings_found

    def forward_chat(
        self,
        input_questions: list[str],
        input_contexts: list[list[str]],
        max_input_length: int = 1024,
        max_output_length: int = 1152,
        num_samples: int = 1,
        temperature: float = 0.7,
        decoding_strategy: str = "temperature",
        stop_strings: list[str] = ["<|eot_id|>", "<|eois_id|>"],
    ) -> tuple[list[str], int, list[dict], list[str], list[bool]]:
        """
        Get individual reasoning steps from the model for batched questions.

        Note: eot_id = 128009, eois_id = 128256

        Args:
            input_questions (list[str]): Input questions for the model.
            input_contexts (list[list[str]]): Previously generated context / steps.
            max_output_length (int): Maximum length of the input. Defaults to 1024.
            max_output_length (int): Maximum length of the generated output. Defaults to 1152.
            num_samples (int): Number of samples to generate. Defaults to 1.
            temperature (float): Temperature for sampling. Defaults to 0.7.
            decoding_strategy (str): Decoding strategy to use. Defaults to "temperature".
            stop_strings (list[str]): List of stop strings to use. Defaults to ["<|eot_id|>", "<|eois_id|>"].
        Returns:
            tuple[list[str], int, list[dict],list[str], list[bool]]: Tuple of the generated texts,
                -1, the prompts, the stop strings found in the generated texts, and whether steps are terminal.
        """
        messages = []
        for input_question, input_context in zip(input_questions, input_contexts):
            if input_context is None:
                messages.append([{"role": "user", "content": input_question}])
            else:
                messages.append(
                    [{"role": "user", "content": input_question}]
                    + [
                        {"role": "intermediate_step", "content": context}
                        for context in input_context
                    ]
                )

        prompts = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=max_input_length,
            add_generation_prompt=True,  # Model training was more stable with this
            return_dict=True,
            return_attention_mask=True,
        ).to(self.device)

        generated_texts, _, stop_strings_found = self.forward(
            prompts,
            max_output_length=max_output_length,
            num_samples=num_samples,
            temperature=temperature,
            stop_strings=stop_strings,
            decoding_strategy=decoding_strategy,
        )

        print(f"Generated texts: {generated_texts}", flush=True)

        # Remove the prompt from the generated text
        for j, prompt in enumerate(input_questions):
            # Remove all special tokens from the prompt
            prompt = self.tokenizer.decode(
                self.tokenizer.apply_chat_template(
                    messages[j],
                    tokenize=True,
                    add_generation_prompt=False,
                    add_special_tokens=False,
                    return_dict=True,
                    return_attention_mask=True,
                ).input_ids,
                skip_special_tokens=True,
            )
            for i in range(num_samples):
                generated_texts[j * num_samples + i] = generated_texts[
                    j * num_samples + i
                ].replace(prompt, "")

        is_final_step = [
            stop_string == "<|eot_id|>" for stop_string in stop_strings_found
        ]

        print(f"Generated texts: {generated_texts}", flush=True)

        print(f"Stop strings found: {stop_strings_found}", flush=True)
        print(f"Is final step: {is_final_step}", flush=True)

        return generated_texts, -1, messages, stop_strings_found, is_final_step

    def forward_prompt(
        self,
        input_prompt: list[str],
        max_input_length: int = 1024,
        max_output_length: int = 1152,
        num_samples: int = 1,
        output_logits: bool = False,
        temperature: float = 0.7,
        decoding_strategy: str = "temperature",
        stop_strings: list[str] = ["<|eot_id|>", "<|eois_id|>"],
    ) -> tuple[list[str], int, list[str]]:
        """
        Tokenize and forward a prompt through the model.

        Note: eot_id = 128009, eois_id = 128256

        Args:
            input_prompt (list[str]): Input prompt.
            max_input_length (int): Maximum length of the input. Defaults to 1024.
            max_output_length (int): Maximum length of the generated output. Defaults to 1152.
            num_samples (int): Number of samples to generate. Defaults to 1.
            output_logits (bool): Whether to output logits. Defaults to False.
            temperature (float): Temperature for sampling. Defaults to 0.7.
            decoding_strategy (str): Decoding strategy to use. Defaults to "temperature".
            stop_strings (list[str]): List of stop strings to use. Defaults to ["<|eot_id|>", "<|eois_id|>"].
        Returns:
            tuple[list[str], int, list[str]]: Tuple of the generated texts, logits if output_logits
                is True, and the stop strings found in the generated texts.
        """
        # Tokenize the input prompt
        inputs = self.tokenizer(
            input_prompt,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=max_input_length,
        ).to(self.device)

        return self.forward(
            inputs,
            max_output_length=max_output_length,
            num_samples=num_samples,
            temperature=temperature,
            stop_strings=stop_strings,
            decoding_strategy=decoding_strategy,
        )

    def get_reasoning_steps(
        self,
        input_questions: list[str],
        input_contexts: list[list[str]],
        max_input_length: int = 1024,
        max_output_length: int = 1152,
        num_samples: int = 1,
        output_logits: bool = False,
        temperature: float = 0.7,
        decoding_strategy: str = "temperature",
    ) -> tuple[list[str], int, list[dict], list[str], list[bool]]:
        """
        Get reasoning steps from the model without the delimiter "assistant" for batched questions.

        Args:
            input_questions (list[str]): Input question for the model.
            input_contexts (list[list[str]]): Previously generated context / steps.
            max_output_length (int): Maximum length of the input. Defaults to 1024.
            max_output_length (int): Maximum length of the generated output. Defaults to 1152.
            num_samples (int): Number of samples to generate. Defaults to 1.
            output_logits (bool): Whether to output logits. Defaults to False.
            temperature (float): Temperature for sampling. Defaults to 0.7.
            decoding_strategy (str): Decoding strategy to use. Defaults to "temperature".
        Returns:
            tuple[list[str], int, list[dict],list[str], list[bool]]: Tuple of the generated texts,
                logits if output_logits is True, the prompts, the stop strings found in the
                generated texts, and whether steps are terminal.
        """
        generated_texts, _, messages, stop_strings_found, is_final_step = (
            self.forward_chat(
                input_questions=input_questions,
                input_contexts=input_contexts,
                max_input_length=max_input_length,
                max_output_length=max_output_length,
                num_samples=num_samples,
                temperature=temperature,
                decoding_strategy=decoding_strategy,
                stop_strings=["<|eot_id|>", "<|eois_id|>"],
            )
        )
        for i in range(len(generated_texts)):
            generated_texts[i] = generated_texts[i].replace("assistant\n\n", "")
        return generated_texts, -1, messages, stop_strings_found, is_final_step

    def simulate_to_end(
        self,
        input_questions: list[str],
        input_contexts: list[list[str]],
        max_input_length: int = 1024,
        max_output_length: int = 1152,
        num_samples: int = 1,
        output_logits: bool = False,
        temperature: float = 0.7,
        decoding_strategy: str = "temperature",
    ) -> tuple[list[str], int, list[dict], list[str], list[bool]]:
        """
        Generate responses till the end of the conversation.

        Args:
            input_questions (list[str]): Input question for the model.
            input_contexts (list[list[str]]): Previously generated context / steps.
            max_output_length (int): Maximum length of the input. Defaults to 1024.
            max_output_length (int): Maximum length of the generated output. Defaults to 1152.
            num_samples (int): Number of samples to generate. Defaults to 1.
            output_logits (bool): Whether to output logits. Defaults to False.
            temperature (float): Temperature for sampling. Defaults to 0.7.
            decoding_strategy (str): Decoding strategy to use. Defaults to "temperature".
        Returns:
            tuple[list[str], int, list[dict],list[str], list[bool]]: Tuple of the generated texts,
                logits if output_logits is True, the prompts, the stop strings found in the
                generated texts, and whether steps are terminal.
        """
        return self.forward_chat(
            input_questions=input_questions,
            input_contexts=input_contexts,
            max_input_length=max_input_length,
            max_output_length=max_output_length,
            num_samples=num_samples,
            temperature=temperature,
            decoding_strategy=decoding_strategy,
            stop_strings=["<|eot_id|>"],
        )

    def merge_lora_layers(self) -> None:
        """
        Merge fine-tuned LoRa layers into the base model.

        Raises:
            ValueError: Raises an exception.
        """
        if hasattr(self.model, "merge_and_unload"):
            self.model = self.model.merge_and_unload()
            print("LoRa layers merged into the base model successfully.")
        else:
            raise ValueError("The current model does not have LoRa layers to merge.")

    def store_model(self, output_path: str = None) -> None:
        """
        Store model in a directory.

        Args:
            output_path (str): Path to output directory. Defaults to None.
        Raises:
            ValueError: Raises an exception.
        """
        if output_path is None:
            output_path = self.save_path
        if output_path is None:
            raise ValueError("No save path specified.")

        self.model.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)
        print(f"Model saved to {output_path}.")

    def ppo_train_on_dataset(
        self,
        samples: list[dict],
        epochs: int = 2,
        learning_rate: float = 1e-5,
        max_length: int = 1024,
    ) -> None:
        """
        PPO training on samples.

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
        processed_data = preprocess_mcts_dataset_policy(
            dataset=samples, tokenizer=self.tokenizer, max_length=max_length
        )
        ppo_trainer = PolicyPPOTrainer(
            policy_model=self.model,
            tokenizer=self.tokenizer,
            device_policy_model=self.device,
            device_old_policy_model="cuda:3",
            summary_writer=self.writer,
            global_step=self.global_step,
        )
        print(
            f"Policy Model GPU summary before training: {torch.cuda.memory_summary(device=self.device)}.",
            flush=True,
        )
        print(
            f"Old Policy Model GPU summary before training: {torch.cuda.memory_summary(device='cuda:3')}.",
            flush=True,
        )
        self.global_step = ppo_trainer.train(
            processed_data,
            num_iterations=epochs,
            train_batch_size=current_app.config["BATCH_SIZE_TRAIN"],
            lr=learning_rate,
        )
        self.writer.flush()

        del processed_data
        del ppo_trainer
        gc.collect()
        torch.cuda.empty_cache()
        print(
            f"Policy Model GPU summary after training: {torch.cuda.memory_summary(device=self.device)}.",
            flush=True,
        )
        print(
            f"Old Policy Model GPU summary after training: {torch.cuda.memory_summary(device='cuda:3')}.",
            flush=True,
        )
        self.store_model(
            output_path=os.path.join(
                self.save_path, "iteration-" + str(self.global_step)
            )
        )
