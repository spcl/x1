# Copyright (c) 2025 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# main author: Eric Schreiber
#
# contributions: Afonso Catarino

from typing import Callable

from x1.synthetic_generation.abstract_verifier import AbstractVerifier

from .utils import _strip_string


def is_a_in_b(a: str, b: str) -> bool:
    """
    Checks if string a is in string b.

    Args:
        a (str): First string.
        b (str): Second string.
    Returns:
        bool: True if a is in b, False otherwise.
    """
    return a in b


def exact_match(a: str, b: str) -> bool:
    """
    Checks if string a is equal to string b.

    Args:
        a (str): First string.
        b (str): Second string.
    Returns:
        bool: True if a is equal to b, False otherwise.
    """
    return a == b


def quasi_match(a: str, b: str) -> bool:
    """
    Checks if string a is equal to string b, considering only numbers and operators.

    Args:
        a (str): First string.
        b (str): Second string.
    Returns:
        bool: True if a is similar to b, False otherwise.
    """

    # remove everything that is not a number or + or - or * or /
    a_only_equation = "".join(
        e for e in a if e.isnumeric() or e in ["+", "-", "*", "/"]
    )
    b_only_equation = "".join(
        e for e in b if e.isnumeric() or e in ["+", "-", "*", "/"]
    )
    if a_only_equation == "" and b_only_equation == "":
        return exact_match(a, b)
    return a_only_equation == b_only_equation


class MATHVerifier(AbstractVerifier):
    """
    The MATHVerifier class verifies tasks and potential solutions within the MATH dataset.
    https://github.com/hendrycks/math

    Inherits from the AbstractVerifier class and implements its abstract methods.
    """

    def __init__(self, compare_fn: Callable[[str, str], bool] = quasi_match) -> None:
        """
        Initialze the MATHVerifier class with the compare function to be used.

        Args:
            compare_fn: (Callable[[str, str], bool]): Function to be used to compare two strings.
                                                      Defaults to quasi_match.
        """
        super().__init__()
        self.compare_fn = compare_fn

    def verify(self, task: dict, output: str) -> int:
        """
        Verify the output of a model for a given task.

        The output is scored with the following scheme:
        0: The output is incorrect.
        2: The problem was solved.

        The evaluation is based on the comparison function compare_fn.

        Args:
            task (dict): Task description as returned by the generator.
            output (str): Model output.
        Returns:
            int: Score for the output.
        """
        answer = _strip_string(task["solution"])
        if output is not None:
            output = _strip_string(output)
            equiv = 2 if self.compare_fn(answer, output) else 0
        else:
            equiv = 0

        return equiv
