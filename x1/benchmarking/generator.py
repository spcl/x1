# Copyright (c) 2025 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# main author: Eric Schreiber
#
# contributions: Afonso Catarino

import os
import time
from typing import Any, Callable, Optional

import requests
from tenacity import retry, wait_fixed

from x1.strategy.mcts import MCTS
from x1.strategy.utils import MinMaxStats, send_queued_server_request

from .abstract_extractor import AbstractExtractor
from .abstract_generator import AbstractGenerator


class StandardGenerator(AbstractGenerator):
    """
    The StandardGenerator class generates a model response for a given input prompt.

    Inherits from the AbstractGenerator class and implements its abstract method.
    """

    def generate_response(self, input_chat: list[dict]) -> str:
        """
        Generate a model response with input_chat as context.

        Args:
            input_chat (list[dict]): Prompt.
        Returns:
            str: Model response.
        """
        tokenized_input = self.tokenizer.apply_chat_template(
            input_chat, tokenize=False, add_generation_prompt=True
        )
        tokenized_input = self.tokenizer(
            tokenized_input, return_tensors="pt", padding="longest", truncation=True
        ).to("cuda")
        output = self.model.generate(
            **tokenized_input,
            max_new_tokens=self.max_new_tokens,
            num_return_sequences=1,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        response = self.tokenizer.batch_decode(
            output, skip_special_tokens=self.skip_special_tokens
        )
        return response[0]


class BoxedGenerator(AbstractGenerator):
    """
    The BoxedGenerator Class generates a model response and reprompts the model to include a
    solution within \\boxed{}.

    Inherits from the AbstractGenerator class and implements its abstract method.
    """

    def generate_response(self, input_chat: list[dict]) -> str:
        """
        Generate a model response with input_chat as context and reprompt the model to include a
        solution within \\boxed{}.

        Args:
            input_chat (list[dict]): Prompt.
        Returns:
            str: Model response.
        """
        tokenized_input = self.tokenizer.apply_chat_template(
            input_chat, tokenize=False, add_generation_prompt=True
        )
        tokenized_input = self.tokenizer(
            tokenized_input, return_tensors="pt", padding="longest", truncation=True
        ).to("cuda")
        output = self.model.generate(
            **tokenized_input,
            max_new_tokens=self.max_new_tokens,
            num_return_sequences=1,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        response = self.tokenizer.batch_decode(
            output, skip_special_tokens=self.skip_special_tokens
        )

        initial_response = response[0]

        # add the response to the chat and ask for a conclusion and box the response
        response = response[0]
        # this is added with the "add_generation_prompt"
        response = response.split("assistant\n\n")[-1]
        input_chat += [
            {"role": "assistant", "content": response},
            {
                "role": "user",
                "content": repr(
                    "Please box your final answer like this: \\boxed{your_final_answer}"
                ),
            },
        ]

        tokenized_input = self.tokenizer.apply_chat_template(
            input_chat, tokenize=False, add_generation_prompt=True
        )
        tokenized_input = self.tokenizer(
            tokenized_input, return_tensors="pt", padding="longest", truncation=True
        ).to("cuda")
        output = self.model.generate(
            **tokenized_input,
            max_new_tokens=self.max_new_tokens,
            num_return_sequences=1,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        response = self.tokenizer.batch_decode(
            output, skip_special_tokens=self.skip_special_tokens
        )

        response = response[0]
        # this is added with the "add_generation_prompt"
        response = response.split("assistant\n\n")[-1]
        return initial_response + " " + response


class MCTSGenerator(AbstractGenerator):
    """
    The MCTSGenerator Class generates a model response by using MCTS.

    Inherits from the AbstractGenerator class and implements its abstract method.
    """

    def __init__(
        self,
        extractor: AbstractExtractor,
        policy_model_url: str = "http://localhost:12440",
        value_model_url: str = "http://localhost:12555",
        max_search_iterations: int = 32,
        min_value: float = -1.0,
        max_value: float = 1.0,
        c1: float = 0,
        c2: float = 19652,
        save_dir: str = "/MCTS_results",
    ) -> None:
        """
        Initialize the MCTSGenerator instance with the configuration and model details.

        Args:
            extractor (AbstractExtractor): Extraction type.
            policy_model_url (str): URL to access the policy model. Defaults to
                                    "http://localhost:12440".
            value_model_url (str): URL to access the value model. Defaults to
                                   "http://localhost:12555".
            max_search_iterations (int): Maximum number of nodes in the search tree. Defaults to 32.
            min_value (float): Minimum value of the value model output. Defaults to -1.0.
            max_value (float): Maximum value of the value model output. Defaults to 1.0.
            c1 (float): Variable to control expansion. Defaults to 0.
            c2 (float): Variable to control expansion. Defaults to 19652.
            save_dir (str): Path to the directory, where the output is stored. Defaults to
                            "/MCTS_results".
        """
        self.is_solution_needed = False

        self.extractor = extractor
        self.max_value = max_value
        self.min_value = min_value
        self.c1 = c1
        self.c2 = c2
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.args = {
            "max_search_iterations": max_search_iterations,
            "policy_api_url": policy_model_url,
            "value_api_url": value_model_url,
        }

        @retry(wait=wait_fixed(4), stop=wait_fixed(30))
        def check_api_url(api_url: str) -> None:
            """
            Check whether the server with the supplied URL is online.

            Args:
                api_url (str): URL of the server.
            Raises:
                ValueError: Raises an exception.
            """
            try:
                response = requests.get(api_url)
                response.raise_for_status()
            except requests.exceptions.RequestException as e:
                raise ValueError(f"Invalid api_url: {e}")

        check_api_url(self.args["policy_api_url"])
        check_api_url(self.args["value_api_url"])
        self.policy_model = LLama3_1_8b_trained(self.args)
        self.value_model = LLama3_1_8b_trained_ValueModel(self.args)

    def generate_response(self, input_chat: list[dict]) -> str:
        """
        Generate a model response using MCTS.

        Args:
            input_chat (list[dict]): Prompt.
        Returns:
            str: Best model response based on the MCTS.
        """

        assert (
            len(input_chat) == 1
        ), "MCTSGenerator only supports single turn conversations (ask one question, get an answer)."

        # Make a new random file in the save_dir
        output_file = os.path.join(self.save_dir, str(time.time()) + ".json")

        mcts = MCTS(self.policy_model, self.value_model, self.args)
        best_path, root = mcts.search_mcts(
            input_chat[0]["content"],
            MinMaxStats(maximum=self.max_value, minimum=self.min_value),
            c1=self.c1,
            c2=self.c2,
        )
        root.save_to_json(output_file)

        if self.extractor.find_solution(best_path[-1]) is not None:
            return best_path[-1]
        else:

            def send_rephrasing_request(question: str, answer: str) -> str:
                """
                Postprocess a model response into a \\boxed{} format by using another model
                interaction.

                Args:
                    question (str): Original task to solve.
                    answer (str): Model response.
                Returns:
                    str: Postprocessed model response into a \\boxed{} format.
                """
                response = send_queued_server_request(
                    {
                        "question": question,
                        "context": [answer, "Please box your final answer."],
                        "num_samples": 1,
                        "prob_entropy_stats": False,
                        "decoding_strategy": "diverse_beam_search",
                    },
                    self.args["policy_api_url"] + "/forward",
                )
                return response["result"][0]

            return send_rephrasing_request(input_chat[0]["content"], best_path[-1])


class MCTSGeneratorWithSimulations(MCTSGenerator):
    """
    The MCTSGeneratorWithSimulations Class that generates a model response by using MCTS with
    simulations of the final solution for evaluation.

    Inherits from the MCTSGenerator Class and overwrites some of its abstract methods.
    """

    def __init__(
        self,
        extractor: AbstractExtractor,
        reward_function: Callable[[Optional[Any], str, list[float]], float],
        policy_model_url: str = "http://localhost:12440",
        value_model_url: str = "http://localhost:12555",
        max_search_iterations: int = 32,
        min_value: float = -1.0,
        max_value: float = 1.0,
        c1: float = 0,
        c2: float = 19652,
        save_dir: str = "/MCTS_results",
        alpha: float = 0,
        num_simulations: int = 8,
    ) -> None:
        """
        Initialize the MCTSGeneratorWithSimulations instance with the configuration and model
        details.

        Args:
            extractor (AbstractExtractor): Extraction type.
            reward function (Callable[[Optional[Any], str, [float]], float]): Function for reward calculation.
            policy_model_url (str): URL to access the policy model. Defaults to
                                    "http://localhost:12440".
            value_model_url (str): URL to access the value model. Defaults to
                                   "http://localhost:12555".
            max_search_iterations (int): Maximum number of nodes in the search tree. Defaults to 32.
            min_value (float): Minimum value of the value model output. Defaults to -1.0.
            max_value (float): Maximum value of the value model output. Defaults to 1.0.
            c1 (float): Variable to control expansion. Defaults to 0.
            c2 (float): Variable to control expansion. Defaults to 19652.
            save_dir (str): Path to the directory, where the output is stored. Defaults to
                            "/MCTS_results".
            alpha (float): Weight of the simulation value. Defaults to 0.
            num_simulations (int): Number of simulations per node. Defaults to 8.
        """

        super().__init__(
            extractor,
            policy_model_url,
            value_model_url,
            max_search_iterations,
            min_value,
            max_value,
            c1,
            c2,
            save_dir,
        )

        self.is_solution_needed = True
        self.alpha = alpha
        self.num_simulations = num_simulations
        self.reward_function = reward_function

        self.args["alpha"] = alpha
        self.args["num_simulations"] = num_simulations
        self.args["reward_function"] = reward_function

    def generate_response(self, input_chat: list[dict], solution: str) -> str:
        """
        Generate a model response using MCTS with simulations of the final solution for evaluation.

        Args:
            input_chat (list[dict]): Prompt.
            solution (str): Ground truth.
        Returns:
            str: Best model response based on the MCTS.
        """
        self.args["gt_solution"] = solution

        return super().generate_response(input_chat)
