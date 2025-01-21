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

import requests
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    wait_random,
)

from .abstract_step_evaluation import AbstractStepEvaluation


class PromptedStepEvaluation(AbstractStepEvaluation):
    """
    Class to evaluate individual reasoning steps with Llama 3.1 70B accessed through the Ollama API.
    The use of this class requires a running Ollama server (see README.md).

    Inherits from the AbstractStepEvaluation class and implements its abstract methods.
    """

    def __init__(self, api_url: str = "http://localhost:11434/api", model: str = "llama3.1:70b") -> None:
        """
        Initialize the PromptedStepEvaluation instance with a configuration.

        Args:
            api_url (str): URL of the Ollama API. Defaults to "http://localhost:11434/api".
            model (str): Model to use for the evaluation. Defaults to "llama3.1:70b".
        """
        self.api_url = api_url
        self.model = model

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10) + wait_random(1, 5),
        retry=retry_if_exception_type(requests.exceptions.RequestException),
    )
    def evaluate_step(self, contextual_memory: list[str], step: str) -> float:
        """
        Evaluate a reasoning step based on the contextual memory.

        Evaluation is tried up to three times with exponential backoff.

        Args:
            contextual_memory (list[str]): List of strings representing the contextual memory.
            step (str): Reasoning step to evaluate.
        Returns:
            float: Value of the reasoning step.
        Raises:
            ValueError: Raises an exception.
        """

        prompt_template = f"""**Task**: {contextual_memory[0]}

**Current Reasoning Step**: {step}

**Previous Reasoning Steps**: {contextual_memory[1:]}

**Instructions**:
You are given a reasoning step as a next step in a reasoning process to solve the given **Task**. Evaluate the **Current Reasoning Step** by considering its overall quality in the context of the problem and previous reasoning steps.
Take into account factors such as relevance to the problem, contribution to progress, and logical consistency.

- Assign an **Overall Score** between **-100 and 100**, where:
- **-100**: Poor quality.
- **0**: Moderate quality.
- **100**: High quality.

**Rules:**
- If the reasoning step contains errors, then it should be scored very low.
- If the reasoning step provides the correct answer to the task, then it should be scored as 100.
- The higher the contribution of the reasoning step to the correct answer, the higher it should be scored.

Provide a brief **justification** for the assigned score.

**Output**:
<analysis>your_reflective_and_strict_analysis</analysis>
<score>your_score_as_integer</score>
"""

        headers = {"Content-Type": "application/json"}
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt_template,
                }
            ],
            "stream": False,
        }

        response = requests.post(self.api_url + "/chat", json=payload, headers=headers)
        response.raise_for_status()

        message = response.json()
        message = message["message"]["content"]

        # Try to check if a score is contained
        score = message.split("<score>")[1].split("</score>")[0]

        try:
            score = int(score)
        except Exception as e:
            raise ValueError(f"Could not evaluate {step}: {e}.")

        return score

    def evaluate_steps(
        self,
        contextual_memory: list[str],
        steps: list[str],
    ) -> list[float]:
        """
        Evaluate multiple reasoning steps based on the contextual memory.

        Args:
            contextual_memory (list[str]): List of strings representing the contextual memory.
            steps (list[str]): Reasoning steps to evaluate.
        Returns:
            list[float]: Values of the reasoning steps.
        """
        scores = []

        for step in steps:
            scores.append(self.evaluate_step(contextual_memory, step))

        return scores

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10) + wait_random(1, 5),
        retry=retry_if_exception_type(requests.exceptions.RequestException),
    )
    def is_solved(self, contextual_memory: list[str], step: str) -> bool:
        """
        Check if the given reasoning steps solves the problem given by the contextual memory.

        The method is tried up to three times with exponential backoff.

        Args:
            contextual_memory (list[str]): List of strings representing the contextual memory.
            step (str): Reasoning step to evaluate for terminality.
        Returns:
            bool: Flag indicating whether the problem is solved.
        """
        prompt_template = f"""Evaluate whether the reasoning step is the **terminal, complete solution** to the problem described in the **Task**.
A reasoning step is terminal if it contains the actual answer to the problem and is not only describing the process to solve the task.
For example, if it is a mathematical problem asking to find the value of x, it must contain the actual number associated with x. If it describes a process, the reasoning step must arrive at a conclusion that directly answers the question.

Output yes if the reasoning step is terminal or no if not inside the <is solved> html tag. Output nothing else.
If the possible solution describes only part of the process or leaves the final step unsolved, mark it as "no". Only mark it as "yes" if the reasoning step provides a complete, conclusive answer to the original question.

**Task**: {contextual_memory[0]}

**Step to evaluate**: {step}

**Previous reasoning steps**: {contextual_memory[1:]}

Output example:
<is solved>yes</is solved>
"""

        headers = {"Content-Type": "application/json"}
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt_template}],
            "stream": False,
        }

        response = requests.post(self.api_url + "/chat", json=payload, headers=headers)
        response.raise_for_status()

        message = response.json()
        message = message["message"]["content"]

        is_solved = message.split("<is solved>")[1].split("</is solved>")[0]

        return is_solved == "yes"
