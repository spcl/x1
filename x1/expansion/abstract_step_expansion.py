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


class AbstractStepExpansion(ABC):
    """
    Abstract base class that defines the interface for the generation of reasoning steps.
    """

    @abstractmethod
    def generate_steps(self, contextual_memory: list[str], num_steps: int) -> list[tuple[str, bool]]:
        """
        Generate the next reasoning steps based on the contextual memory.

        Args:
            contextual_memory (list[str]): List of strings that represent the context.
            num_steps (int): Number of reasoning steps to generate.
        Returns:
            list[tuple[str, bool]]: List of tuples containing the reasoning steps and a boolean
                                    indicating if it's the final reasoning step
        """
        pass

    @abstractmethod
    def simulate_end(self, contextual_memory: list[str], num_simulations: int) -> list[str]:
        """
        Simulate the end of the reasoning based on the contextual memory.

        Args:
            contextual_memory (list[str]): List of strings that represent the context.
            num_simulations (int): Number of simulations to run.
        Returns:
            list[str]: List of final answers.
        """
        pass
