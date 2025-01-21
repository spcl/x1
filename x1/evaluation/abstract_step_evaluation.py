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

from abc import ABC, abstractmethod


class AbstractStepEvaluation(ABC):
    """
    Abstract class that defines the interface for the evaluation of reasoning steps.
    """

    @abstractmethod
    def evaluate_step(self, contextual_memory: list[str], step: str) -> float:
        """
        Evaluate a reasoning step based on the contextual memory.

        Args:
            contextual_memory (list[str]): List of strings representing the contextual memory.
            step (str): Reasoning step to evaluate.
        Returns:
            float: Value of the reasoning step.
        """
        pass

    @abstractmethod
    def evaluate_steps(
        self, contextual_memory: list[str], steps: list[str]
    ) -> list[float]:
        """
        Evaluate multiple reasoning steps based on the contextual memory.

        Args:
            contextual_memory (list[str]): List of strings representing the contextual memory.
            steps (list[str]): Reasoning steps to evaluate.
        Returns:
            list[float]: Values of the reasoning steps.
        """
        pass
