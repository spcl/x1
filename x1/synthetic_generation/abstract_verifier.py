# Copyright (c) 2025 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# main author: Afonso Catarino

from abc import ABC, abstractmethod


class AbstractVerifier(ABC):
    """
    Abstract base class that defines the interface for all domain specific verifiers.
    """

    @abstractmethod
    def verify(self, task: dict, output: str) -> int:
        """
        Verify the output of a model for a given task.

        The output is scored with the following scheme:
        -1: The output is invalid.
         0: The output is incorrect.
         1: The output satisfies the constraints of the problem. It may lead to a correct or
            an incorrect solution.
         2: The problem was solved.

        Not all scores must be implemented, but at least 0 and 2 are advised.

        Args:
            task (dict): Task description as returned by the generator.
            output (str): Model output.
        Returns:
            int: Score for the output.
        """
        pass
