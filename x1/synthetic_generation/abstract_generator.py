# Copyright (c) 2025 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# main author: Afonso Catarino
#
# contributions: Hannes Eberhard

from abc import ABC, abstractmethod
from random import Random


class AbstractGenerator(ABC):
    """
    Abstract base class that defines the interface for all domain specific generators.
    """

    random: Random

    def __init__(self, seed: int | None = None) -> None:
        """
        Initialize the AbstractGenerator instance with a seed for the random number generation.

        Args:
            seed (int | None): Seed for the random number generation. Defaults to None.
        """
        self.random = Random(seed)

    @abstractmethod
    def generate(self, difficulty: float) -> dict:
        """
        Generate a task with a specific difficulty.

        Output format:
        {
            "type": str, # task domain
            "level": int, # difficulty level in [1, 10]
            "problem": str, # task description
            "unique_solution": bool, # whether the task has a unique solution
            "solution": str, # solution to the task
        }

        Args:
            difficulty (float): Difficulty level in [0, 1].
        Returns:
            dict: Task description.
        """
        pass
