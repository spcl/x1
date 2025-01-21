# Copyright (c) 2025 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# main author: Afonso Catarino

import re
from dataclasses import dataclass, field

from .abstract_generator import AbstractGenerator
from .abstract_verifier import AbstractVerifier


@dataclass
class Thing:
    """
    Data class to represent the different properties of a object.
    """

    weight: int = 0
    value: int = 0


@dataclass
class Game:
    """
    Data class to represent the state of a game.
    """

    weight_limit: int = 0
    value_target: int = 0
    things: dict[str, Thing] = field(default_factory=dict)


class KnapsackGenerator(AbstractGenerator):
    """
    The KnapsackGenerator class generates tasks within the knapsack domain.

    Inherits from the AbstractGenerator class and implements its abstract methods.
    """

    objects = [
        "pen",
        "map",
        "book",
        "flask",
        "snack",
        "rope",
        "brush",
        "towel",
        "knife",
        "band",
        "mask",
        "soap",
        "torch",
        "screw",
        "coin",
        "key",
        "lock",
        "lens",
        "card",
        "note",
        "cable",
        "tape",
        "flash",
        "cloth",
        "case",
        "file",
        "snack",
        "cap",
        "clip",
        "tag",
    ]

    def __init__(self, seed: int | None = None) -> None:
        """
        Initialize the KnapsackGenerator instance with a seed for the random number generation.

        Args:
            seed (int | None): Seed for the random number generation. Defaults to None.
        """
        super().__init__(seed=seed)

    def generate(self, difficulty: float) -> str:
        """
        Generate a task in the knapsack domain.

        The difficulty determines the number of objects to consider as well as the difference of
        the value target to the highest possible value.

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

        num_things = 3 + int(difficulty * (len(self.objects) - 3))
        thing_weight_bound = 20
        thing_value_bound = 20

        things = dict()
        for i in range(num_things):
            things[self.objects[i]] = Thing(
                self.random.randint(1, thing_weight_bound),
                self.random.randint(1, thing_value_bound),
            )

        weight_limit = self.random.randint(
            int(num_things * thing_weight_bound / 4),
            int(3 * num_things * thing_weight_bound / 4),
        )

        game = Game(weight_limit=weight_limit, value_target=0, things=things)

        max_value = self._solve(game)
        value_target = self.random.randint(
            int(max_value - (1 - difficulty + 0.1) * thing_value_bound),
            int(max_value + (1 - difficulty + 0.1) * thing_value_bound),
        )
        game.value_target = value_target

        problem = f"Imagine you're packing for a trip. You are going to carry your backpack for a long time, so you don't want its weight to exceed {game.weight_limit + 1}kg. The backpack itself weighs 1kg. Each item you are considering to pack has a weight and a value. You want the items in your backpack to amount to the highest possible value while maintaining the weight of the backpack below the aforementioned threshold. You are considering the following items and can only pack one of each:\n\n"
        for thing in things:
            problem += f"{thing}\tweight: {things[thing].weight}\tvalue: {things[thing].value}\n"
        problem += f"\nTrue or False: There is a list of items whose value sum is higher than {game.value_target} and weight sum is lower than {game.weight_limit}."

        task = {}
        task["type"] = "Knapsack"
        task["level"] = int(difficulty * 10) + 1
        task["problem"] = problem
        task["unique_solution"] = True
        task["solution"] = max_value >= game.value_target

        return task

    def _solve(self, game: Game) -> int:
        """
        Solve a knapsack problem.

        Args:
            game (Game): Task setting.
        Returns:
            int: Maximum value that can be obtained.
        """

        n = len(game.things)
        ind = [thing for thing in game.things]

        dp = [[0] * (game.weight_limit + 1) for _ in range(n + 1)]

        for i in range(1, n + 1):
            for w in range(game.weight_limit + 1):
                if game.things[ind[i - 1]].weight > w:
                    dp[i][w] = dp[i - 1][w]
                else:
                    dp[i][w] = max(
                        dp[i - 1][w],
                        dp[i - 1][w - game.things[ind[i - 1]].weight]
                        + game.things[ind[i - 1]].value,
                    )

        max_value = dp[n][game.weight_limit]

        return max_value


class KnapsackVerifier(AbstractVerifier):
    """
    The KnapsackVerifier class verifies tasks and potential solutions within the knapsack domain.

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
