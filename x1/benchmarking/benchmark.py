# Copyright (c) 2025 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# main author: Eric Schreiber
#
# contributions: Afonso Catarino

import json
import os

from tqdm import tqdm

from x1.synthetic_generation import AbstractVerifier

from .abstract_extractor import AbstractExtractor
from .abstract_generator import AbstractGenerator


def return_template(user_input: str) -> list[dict]:
    """
    Return a template for the user input.

    Args:
        user_input (str): User input.
    Returns:
        list[dict]: Template for the user input.
    """
    return [
        {
            "role": "user",
            "content": f"{user_input}",
        }
    ]


def compare_answers(
    output: str,
    ground_truth: str,
    extractor: AbstractExtractor,
    verifier: AbstractVerifier,
) -> tuple[bool, str, str]:
    """
    Extract and verify an answer based on user-defined extractor and verifier classes.

    Args:
        output (str): Model response.
        ground_truth (str): Ground truth.
        extractor (AbstractExtractor): Extraction type.
        verifier (AbstractVerifier): Domain verifier.
    Returns:
        tuple[bool, str, str]: The first element is True if the output and the ground truth are
                               equivalent, False otherwise. The second element is the extracted
                               output and the third element is the extracted ground truth.
    """
    extracted_output = extractor.find_solution(output)
    extracted_ground_truth = extractor.find_solution(ground_truth)
    equiv = (
        verifier.verify(task={"solution": ground_truth}, output=extracted_output) == 2
    )
    return equiv, extracted_output, extracted_ground_truth


def generate_answers(
    dataset: list[dict], generator: AbstractGenerator, output_file_path: str
) -> None:
    """
    Generate model responses for a dataset of tasks and store these responses in a JSON file.

    Args:
        dataset (list[dict]): Dataset of tasks.
        generator (AbstractGenerator): Generation type.
        output_file_path (str): Path to output file.
    Raises:
        ValueError: Raises an exception.
    """
    assert output_file_path is not None, "Please provide an output path."
    # create the file if it does not exist
    if not os.path.exists(output_file_path):
        with open(f"{output_file_path}", "w") as f:
            f.write("[")
    else:
        raise ValueError("Output path already exists. Please provide a new path.")

    for i, example in tqdm(enumerate(dataset)):
        output_str = generator.generate_response(return_template(example["problem"]))
        with open(f"{output_file_path}", "a") as f:
            json.dump(example | {"output": output_str}, f)
            if i < len(dataset) - 1:
                f.write(",\n")
    with open(f"{output_file_path}", "a") as f:
        f.write("]")


def extract_answers(
    extractor: AbstractExtractor, verifier: AbstractVerifier, file_path: str
) -> float:
    """
    Extract answers from the generated responses in a file and store the extracted answers in the
    same file.

    Args:
        extractor (AbstractExtractor): Extraction type.
        verifier (AbstractVerifier): Domain verifier.
        file_path (str): Path to file.
    Returns:
        float: Aggregated accuracy for the dataset in the file.
    """
    assert file_path is not None, "Please provide a file path."
    with open(file_path, "r") as f:
        answers = json.load(f)

    # FOR LEGACY SUPPORT
    # if answers is a dict, convert it to a list
    if isinstance(answers, dict):
        answers_list = []
        for key in answers:
            answers_list.append(answers[key])
        answers = answers_list

    for i, example in tqdm(enumerate(answers)):
        output_str = example["output"]
        assert "solution" in example, "Please provide the solution in the dataset."
        gt_string = example["solution"]
        equiv, extracted_output, extracted_answer = compare_answers(
            output_str, gt_string, extractor, verifier
        )
        answers[i]["extracted_output"] = extracted_output
        answers[i]["extracted_answer"] = extracted_answer
        answers[i]["correct"] = bool(equiv)

    with open(file_path, "w") as f:
        json.dump(answers, f)

    return sum([1 for example in answers if example["correct"]]) / len(answers)


def compute_benchmark(
    dataset: list[dict],
    generator: AbstractGenerator,
    extractor: AbstractExtractor,
    verifier: AbstractVerifier,
    output_file_path: str,
) -> float:
    """
    Execute a benchmark by first generating reponses for a dataset of tasks with a subsequent
    extraction and verification of the actual answers from the model responses, storing the results
    in an output file, and returning the aggregated accuracy for the model on the tasks of the
    dataset.

    Args:
        dataset (list[dict]): Dataset of tasks.
        generator (AbstractGenerator): Generation type.
        extractor (AbstractExtractor): Extraction type.
        verifier (AbstractVerifier): Domain verifier.
        output_file_path (str): Path to output file.
    Returns:
        float: Aggregated accuracy of the model.
    """
    assert not os.path.exists(
        output_file_path
    ), "Output path already exists. Please provide a new path."

    generate_answers(dataset, generator, output_file_path)
    accuracy = extract_answers(extractor, verifier, output_file_path)
    return accuracy
