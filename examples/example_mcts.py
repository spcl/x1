# Copyright (c) 2025 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# main author: Eric Schreiber
#
# contributions: Afonso Catarino

"""
This example code uses MCTS with a policy model, but no value model, to solve a single task from
the MATH dataset.
"""

import requests

from x1.benchmarking import MATHExtractor, MATHVerifier, compare_answers, quasi_match
from x1.expansion import PolicyModel
from x1.strategy import MCTS

# Initialize extractor and verifier
extractor = MATHExtractor()
verifier = MATHVerifier(quasi_match)


def check_api_url(api_url: str) -> None:
    """
    Check if the API URL is valid.

    Args:
        api_url (str): API URL to check.
    Raises:
        ValueError: Raises an exception.
    """
    try:
        response = requests.get(api_url)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"Own server is not running.", flush=True)
        raise ValueError(f"Invalid api_url: {e}")


def reward_function(
    output_str: str, gt_string: str, reward_range: list[float]
) -> float:
    """
    Reward function for the MCTS.

    Args:
        output_str (str): Output string from the model.
        gt_string (str): Ground truth string.
        reward_range (list[float]): Rewards associated with failing and succeeding.
    Returns:
        float: Reward value.
    """
    if not output_str:
        print("Output string is empty.", flush=True)
        return reward_range[0]

    equiv, extracted_output, extracted_answer = compare_answers(
        output_str, gt_string, extractor, verifier
    )

    # Scale the boolean reward to the reward range
    reward = reward_range[1] if equiv else reward_range[0]

    return reward


if __name__ == "__main__":
    args = {
        "alpha": 1,  # Take only the value of the simulations. Nothing from the value model.
        "c1": 0.1,
        "c2": 19652,
        "gt_solution": "For the product of two factors to be negative, one of the factors must be positive and one of the factors must be negative. Since $x-5<x+5$ for all $x$, we know that $x$ satisfies $(x-5)(x+5)<0$ if and only if $$x-5 < 0 < x+5.$$ The first inequality, $x-5<0$, tells us that $x<5$. The second inequality, $0<x+5$, tells us that $x>-5$. Thus the solutions to the original inequality are given by $-5<x<5$.\n\nThe smallest integer in this interval is $x=\\boxed{-4}$.",
        "max_search_iterations": 4,
        "num_simulations_per_node": 4,
        "output_path": "graphs",
        "policy_api_url": "http://localhost:12440",
        "question": "Find the smallest integer that satisfies the inequality: \\[\n(x-5)(x+5)<0.\n\\]",
        "reward_function": reward_function,
        "reward_range": [-1, 1],
        "save_tree": True,
    }

    print("Starting MCTS *** Just checking your API connections now ***")
    check_api_url(args["policy_api_url"])

    policy_model = PolicyModel(args["policy_api_url"])
    mcts = MCTS(expansion=policy_model, evaluation=None, args=args)
    best_path, root = mcts.run_mcts(args)

    print("MCTS search completed")
    print(f"Best path: {best_path}")
    print(f"Value: {root.q_value}")
    print(f"Solution found: {best_path[-1]}")
