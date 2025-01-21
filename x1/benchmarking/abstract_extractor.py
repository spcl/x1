# Copyright (c) 2025 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# main author: Eric Schreiber
#
# contributions: Afonso Catarino

from abc import ABC, abstractmethod


class AbstractExtractor(ABC):
    """
    Abstract base class that defines the interface for extracting answers from model responses.
    """

    @abstractmethod
    def find_solution(self, response: str) -> str | None:
        """
        Extract solution from model response.

        Args:
            response (str): Model response.
        Returns:
            str: Solution if found, else None.
        """
        pass
