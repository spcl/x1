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

import re

import requests
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    wait_random,
)

from .abstract_step_expansion import AbstractStepExpansion


class PromptedStepExpansion(AbstractStepExpansion):
    """
    Class to generate reasoning steps with Llama 3.1 70B accessed through the Ollama API.
    The use of this class requires a running Ollama server (see README.md).

    Inherits from the AbstractStepExpansion class and implements its abstract methods.
    """

    def __init__(self, api_url: str = "http://localhost:11434/api", model: str = "llama3.1:70b") -> None:
        """
        Initialize the PromptedStepExpansion instance with a configuration.

        Args:
            api_url (str): URL of the Ollama API. Defaults to "http://localhost:11434/api".
            model (str): Model to use for the expansion. Defaults to "llama3.1:70b".
        """
        self.api_url = api_url
        self.model = model

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10) + wait_random(1, 5),
        retry=retry_if_exception_type(requests.exceptions.RequestException),
    )
    def generate_steps(self, contextual_memory: list[str], num_steps: int) -> list[tuple[str, bool]]:
        """
        Generate the next reasoning steps based on the contextual memory.

        Generation is tried up to three times with exponential backoff.

        Args:
            contextual_memory (list[str]): List of strings that represent the context.
            num_steps (int): Number of steps to generate.
        Returns:
            list[tuple[str, bool]]: List of tuples containing the reasoning steps and a boolean
                                    indicating if it's the final reasoning step.
        """
        # Create prompt
        prompt_template = f"""<question>{contextual_memory[0]}</question>
<chain of steps so far>{contextual_memory[1:]}</chain of steps so far>

You are tasked with generating the next step in the reasoning process to help solve the given question.
Based on the chain of steps so far, provide a list of possible next reasoning step candidates that logically follow from the current reasoning as the next reasoning step.

Please ensure each reasoning step is:
- Be **independent** from the other reasoning steps in the list. **Do not** build any reasoning step based on another reasoning step in the list.
- Be **directly connected** to the chain of steps so far, continuing from the current reasoning.
- **Progress toward solving the problem**.
- Be **diverse**, offering different approaches or angles to continue reasoning.
- **Do not only describe what should be done**, but also **take action** within the reasoning step.

Provide {num_steps} distinct next steps for the reasoning process. If a step contains the final solution, include <final> at the end of the step.

Response format for 3 steps:
<step>first_possible_next_step</step>
<step>second_possible_next_step</step>
<step>third_possible_next_step</step>"""

        headers = {"Content-Type": "application/json"}
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt_template}],
            "stream": False,
        }

        # Get request
        response = requests.post(
            self.api_url + "/chat", headers=headers, json=payload, timeout=20
        )

        # Error handling
        response.raise_for_status()

        # extract reasoning steps from response
        message = response.json()
        message = str(message["message"]["content"])

        steps = message.split("</step>")[
            :-1
        ]  # The last one is either empty or indistinct writing

        steps = [step.split("<step>")[1] for step in steps]
        steps = [
            (step.replace("<final>", "").strip(), "<final>" in step) for step in steps
        ]

        return steps

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10) + wait_random(1, 5),
        retry=retry_if_exception_type(requests.exceptions.RequestException),
    )
    def simulate_end(self, contextual_memory: list[str], num_simulations: int) -> list[str]:
        """
        Simulate the end of the reasoning chain based on the contextual memory starting from 'step'.

        Generation is tried up to three times with exponential backoff.

        Args:
            contextual_memory (list[str]): List of strings that represent the context.
            num_simulations (int): Number of simulations to run.
        Returns:
            list[str]: List of final answers.
        """
        # Create prompt
        template = f"""<question>{contextual_memory[0]}</question>
<past_chain>{contextual_memory[1:]}</past_chain>

You are given a question and a chain of steps. The chain of steps contains the past chain and the reasoning step that is the starting point for your task.
Your task is to simulate the reasoning process after the step in the reasoning chain to solve the question.
- **Do not ignore the given reasoning chain or reasoning step** by solving the question directly. Your simulation is used to evaluate the reasoning step so its capabilities are tested through your simulation.
- **Follow logically** from the current state, reasoning step-by-step until you reach a terminal conclusion.
- Ensure that every intermediate step is logically consistent and that the reasoning is **free from errors**.
- The goal is to explore whether you can logically progress to a final solution based on the current context.
- You need to extract the final result from your process that is requested in the question based on the given reasoning process.

Make sure to not introduce any mistakes in the reasoning process, and output a solution only if it logically follows from the provided chain of steps.
For the output, first provide the reasoning chain till the end in the <simulation> tags. Then output the final result always into <answer> tags in the last line. Do not use the <answer> tag anywhere than in the last line for the final answer of your simulation.

**Output format:**
<simulation>reasoning_chain_here</simulation>
<answer>final_answer_here</answer>"""

        headers = {"Content-Type": "application/json"}
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": template}],
            "stream": False,
        }

        answers = []
        
        for _ in range(num_simulations):
            # Get request
            response = requests.post(
                self.api_url + "/chat", headers=headers, json=payload, timeout=20
            )

            # Error handling
            response.raise_for_status()

            # extract the answer from the message - a common error of the LLM is returning answers in the boxed syntax
            message = response.json()
            message = message["message"]["content"]

            if "<answer>" in message and "</answer>" in message:
                try:
                    message = message.split("</answer>")[0].split("<answer>")[1]
                except IndexError:
                    print("IndexError: Problem splitting message. Message: {message}")
            else:
                try:
                    match = re.search(r"\\boxed{([^}]*)}", message)
                    if match:
                        message = match.group(1)  # Return the boxed content
                except:
                    print(f"Tags <answer> or </answer> not found in the message: {message}")
                    message = None
            answers.append(message)

        return answers
