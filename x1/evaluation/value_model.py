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

import requests
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    wait_random,
)

from x1.strategy.utils import send_request

from .abstract_step_evaluation import AbstractStepEvaluation


class ValueModel(AbstractStepEvaluation):
    """
    Class to evaluate individual reasoning steps with a value model, that is hosted within the
    server environment

    Inherits from the AbstractStepEvaluation class and implements its abstract methods.
    """

    def __init__(self, value_api_url: str) -> None:
        """
        Initialize the ValueModel instance with a configuration.

        Args:
            value_api_url (str): URL of the value server.
        """
        self.value_api_url = value_api_url

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10) + wait_random(1, 5),
        retry=retry_if_exception_type(requests.exceptions.RequestException),
    )
    def evaluate_step(
        self, contextual_memory: list[str], step: str, is_final: bool
    ) -> float:
        """
        Evaluate a reasoning step based on the contextual memory.

        Evaluation is tried up to three times with exponential backoff.

        Args:
            contextual_memory (list[str]): List of strings representing the contextual memory.
            step (str): Reasoning step to evaluate.
            is_final (bool): Flag indicating whether the problem is solved.
        Returns:
            float: Value of the reasoning step.
        Raises:
            ValueError: Raises an exception.
        """
        # Check that we always have a question and a reasoning step
        if len(contextual_memory) == 0:
            raise ValueError(
                "Contextual memory should contain at least one element: the question."
            )
        if len(step) == 0 or step is None:
            print(f"WARNING: step is empty for the context {contextual_memory}")

        messages = [{"role": "user", "content": contextual_memory[0]}]
        length = len(contextual_memory)
        for i in range(1, length):
            messages.append(
                {"role": "intermediate_step", "content": contextual_memory[i]}
            )

        messages.append(
            {
                "role": f"{'last_step' if is_final else 'intermediate_step'}",
                "content": step,
            }
        )

        response = send_request(
            api_url=self.value_api_url + "/forward", payload={"messages": messages}
        )

        return response["result"][0]

    def evaluate_steps(
        self, contextual_memory: list[str], steps: list[str], is_final: list[bool]
    ) -> list[float]:
        """
        Evaluate multiple reasoning steps based on the contextual memory.

        Args:
            contextual_memory (list[str]): List of strings representing the contextual memory.
            steps (list[str]): Reasoning steps to evaluate.
            is_final (list[bool]): List of flags indicating whether the respective step is final.
        Returns:
            list[float]: Values of the reasoning steps.
        """
        values = []
        for i, step in enumerate(steps):
            values.append(self.evaluate_step(contextual_memory, step, is_final[i]))
        return values
