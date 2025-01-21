# Copyright (c) 2025 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# main author: Eric Schreiber
#
# contributions: Afonso Catarino
#
# The MATHExtractor code uses small portions of the GSM8K benchmark code
# (https://github.com/openai/grade-school-math), which is indicated at
# the respective places. The GSM8K code has the following license:
#
# MIT License
#
# Copyright (c) 2021 OpenAI
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import re

from .abstract_extractor import AbstractExtractor
from .utils import last_boxed_only_string, last_solution_only_string, remove_boxed


class DummyExtractor(AbstractExtractor):
    """
    Extractor that returns the full response.
    """

    def find_solution(self, response: str) -> str:
        """
        Extract the full response.

        Args:
            response (str): Model response.
        Returns:
            str: Full response.
        """
        return response


class MATHExtractor(AbstractExtractor):
    """
    Extractor for the MATH dataset.
    """

    def __init__(self):
        """
        Initialize the MATHExtractor instance with the regular expression pattern.

        The code is copied from the GSM8K benchmark:
        https://github.com/openai/grade-school-math/blob/master/grade_school_math/dataset.py#L24
        """
        self.ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")

    def find_solution_boxed(self, response: str) -> str | None:
        """
        Extract only what is within \\boxed{}.
        Analogous to what is done in the MATH benchmark.

        Args:
            response (str): Model response.
        Returns:
            str: Text within \\boxed{}.
        """
        response = last_boxed_only_string(response)
        return remove_boxed(response)

    def find_solution_answer(self, response: str) -> str | None:
        """
        Extract only what is after "The answer is ".

        Args:
            response (str): Model response.
        Returns:
            str: Text after "The answer is ".
        """
        response = last_solution_only_string(response)
        return response

    def find_solution_cardinal(self, response: str) -> str | None:
        """
        Extract only what is after "####".

        The code is copied from the GSM8K benchmark:
        https://github.com/openai/grade-school-math/blob/master/grade_school_math/dataset.py#L28

        Args:
            response (str): Model response.
        Returns:
            str: Text after "####".
        """
        # Stripping of everything before the last "####"
        match = self.ANS_RE.search(response)
        if match:
            match_str = match.group(1).strip()
            match_str = match_str.replace(",", "")
            return match_str
        return None

    def find_solution(self, response: str) -> str | None:
        """
        Extract the solution within \\boxed{}, after "####" or after "The answer is ".

        Args:
            response (str): Model response.
        Returns:
            str: Solution.
        """
        # run through all three methods
        methods = [
            self.find_solution_boxed,
            self.find_solution_answer,
            self.find_solution_cardinal,
        ]
        for method in methods:
            solution = method(response)
            if solution is not None:
                return solution

        return None
