# Copyright (c) 2025 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# main author: Afonso Catarino
#
# contributions: Hannes Eberhard
#
# This code uses a slightly modified copy of the PrOntoQA framework to generate
# reasoning tasks based on logic, which can be found in the PrOntoQA.
#
# The original PrOntoQA framework can be accessed at
# https://github.com/asaparov/prontoqa
# Its license can be found in PrOntoQA/LICENSE.

import re

from .abstract_generator import AbstractGenerator
from .abstract_verifier import AbstractVerifier
from .PrOntoQA import random
from .PrOntoQA.run_experiment import generate_question


class PrOntoQAGenerator(AbstractGenerator):
    """
    The PrOntoQAGenerator class generates tasks within the PrOntoQA domain.

    Inherits from the AbstractGenerator class and implements its abstract methods.
    """

    def __init__(self, seed: int | None = None) -> None:
        """
        Initialize the PrOntoQAGenerator instance with a seed for the random number generation.

        Args:
            seed (int | None): Seed for the random number generation. Defaults to None.
        """
        super().__init__(seed=seed)
        random.seed(seed)

    def generate(self, difficulty: float) -> dict:
        """
        Generate a task in the ProntoQA domain.

        The difficulty determines the number of deduction steps in the required proof.

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

        num_deduction_steps = 2 + int(difficulty * 5)

        self.question = None
        while self.question is None:
            (
                self.question,
                self.query,
                self.axioms,
                self.chain_of_thought,
                self.answer,
                self.proof,
            ) = generate_question(
                num_deduction_steps=num_deduction_steps,
                available_concept_names=None,
                deduction_rule="ModusPonens",
                proofs_only=False,
            )

        problem = "Q: " + str(self.question) + "\n" + str(self.query) + "\n\nA: "

        task = {}
        task["type"] = "ProntoQA"
        task["level"] = int(difficulty * 10) + 1
        task["problem"] = problem
        task["unique_solution"] = True
        task["solution"] = self.answer

        return task


class PrOntoQAVerifier(AbstractVerifier):
    """
    The PrOntoQAVerifier class verifies tasks and potential solutions within the PrOntoQA domain.

    Inherits from the AbstractVerifier class and implements its abstract methods.
    """

    def verify(self, task: dict, output: str) -> int:
        """
        Verify the output of a model for a given task.

        The output is scored with the following scheme:
        0: The output is incorrect.
        2: The problem was solved.

        The scoring is based on the last occurrence of True or False in the output.

        Args:
            task (dict): Task description as returned by the generator.
            output (str): Model output.
        Returns:
            int: Score for the output.
        """

        pattern = r"(true|false)"
        matches = re.findall(pattern, output, re.IGNORECASE)
        correct = (
            matches[-1].lower() == str(task["solution"]).lower()
            if len(matches) > 0
            else False
        )
        return correct
