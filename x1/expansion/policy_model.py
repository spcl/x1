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

from typing import Any

import requests
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    wait_random,
)

from x1.strategy.utils import send_queued_server_request

from .abstract_step_expansion import AbstractStepExpansion


class PolicyModel(AbstractStepExpansion):
    """
    Class to generate individual reasoning steps with a policy model, that is hosted within the
    server environment.

    Inherits from the AbstractStepExpansion class and implements its abstract methods.
    """

    def __init__(self, policy_api_url: str) -> None:
        """
        Initialize the PolicyModel instance with a configuration.

        Args:
            policy_api_url (str): URL of the policy server.
        """
        self.policy_api_url = policy_api_url

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10) + wait_random(1, 5),
        retry=retry_if_exception_type(requests.exceptions.RequestException),
    )
    def generate_steps(
        self,
        contextual_memory: list[str],
        num_steps: int,
    ) -> list[tuple[str, bool]]:
        """
        Generate the next reasoning steps based on the contextual memory.

        Generation is tried up to three times with exponential backoff.

        Args:
            contextual_memory (list[str]): List of strings that represent the context.
            num_steps (int): Number of steps to generate.
        Returns:
            list[tuple[str, bool]]: List of tuples containing the reasoning steps and a boolean
                                    indicating if it's the final reasoning step.
        """
        response = send_queued_server_request(
            {
                "question": contextual_memory[0],
                "context": (
                    None if len(contextual_memory) == 1 else contextual_memory[1:]
                ),
                "num_samples": num_steps,
                "decoding_strategy": "diverse_beam_search",
            },
            self.policy_api_url + "/forward",
        )
        return_list = [
            (step, is_final)
            for step, is_final in zip(response["result"], response["is_final_step"])
        ]
        return return_list

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10) + wait_random(1, 5),
        retry=retry_if_exception_type(requests.exceptions.RequestException),
    )
    def simulate_end(self, contextual_memory: list[str], num_simulations: int) -> list[str]:
        """
        Simulate the end of the reasoning based on the contextual memory.

        Generation is tried up to three times with exponential backoff.

        Args:
            contextual_memory (list[str]): List of strings that represent the context.
            num_simulations (int): Number of simulations to run on the server.
        Returns:
            list[str]: List of final answers.
        """
        response = send_queued_server_request(
            {
                "question": contextual_memory[0],
                "context": contextual_memory[1:],
                "num_samples": num_simulations,
                "full_simulation": True,
                "decoding_strategy": "diverse_beam_search",
            },
            self.policy_api_url + "/forward",
        )
        return response["result"]
