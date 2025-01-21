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

import os
import time
import uuid
from threading import Thread

from flask import Response, current_app, jsonify, request
from flask.ctx import AppContext


# GET: status of the server
@current_app.route("/", methods=["GET"])
def status() -> tuple[Response, int]:
    """
    When GET request is made to /:
    Return the status of the server.

    Returns:
        tuple[Response, int]: Tuple of the response and the status code.
    """
    return (
        jsonify(
            {
                "status": f"Server is running, ALLOW_TRAINING: {current_app.config['ALLOW_TRAINING']}, STOP_TRAINING_FLAG: {current_app.config['STOP_TRAINING_FLAG']}",
                "allow_training": current_app.config["ALLOW_TRAINING"],
                "stop_training_flag": current_app.config["STOP_TRAINING_FLAG"],
                "allow_inference": current_app.config["ALLOW_TRAINING"],
            }
        ),
        200,
    )


# GET: Get the result of a forward pass
@current_app.route("/forward/<task_id>", methods=["GET"])
def get_result(task_id: str) -> tuple[Response, int]:
    """
    When GET request is made to /forward/<task_id>:
    Return the model output to the input determined by task_id. If task_id is not found (either
    because the requested task_id was never generated, inference has not finished or the result
    was already retrieved), return an error.

    Args:
        task_id (str): Id for the task to retrieve the model output for.
    Returns:
        tuple[Response, int]: Tuple of the response and the status code.
    """
    if task_id not in current_app.config["INFERENCE_FINISHED_DICT"]:
        return (
            jsonify(
                {"error": "Task not found (maybe not finished or already retrieved)."}
            ),
            404,
        )

    # Remove the result from the dictionary
    result = current_app.config["INFERENCE_FINISHED_DICT"].pop(task_id)
    return jsonify(result), 200


# POST: Forward pass on input data
@current_app.route("/forward", methods=["POST"])
def forward() -> tuple[Response, int]:
    """
    When POST request is made to /forward:
    Append a task with the payload's 'question', 'safe_logits', 'decoding_strategy' and
    'full_simulation' fields to the inference queue and return task_id. If the input is
    invalid, return an error.

    Returns:
        tuple[Response, int]: Tuple of the response and the status code.
    """
    data = request.json
    input_question = data.get("question", None)
    safe_logits = data.get("safe_logits", False)
    decoding_strategy = data.get("decoding_strategy", "temperature")
    simulate_to_end = data.get("full_simulation", False)

    if not input_question:
        return jsonify({"error": "Invalid input"}), 400

    if safe_logits:
        return (
            jsonify({"error": "safe_logits is not implemented for forward"}),
            501,
        )

    if decoding_strategy not in ["temperature", "diverse_beam_search", "beam_search"]:
        return (
            jsonify(
                {
                    "error": "Invalid decoding strategy. Supported strategies are 'temperature', 'diverse_beam_search', 'beam_search'."
                }
            ),
            400,
        )

    if not isinstance(simulate_to_end, bool):
        return jsonify({"error": "full_simulation must be a boolean"}), 400

    # Add to queue
    time_now = time.time()
    task_id = str(uuid.uuid4())
    if simulate_to_end:
        current_app.config["INFERENCE_SIMULATION_QUEUE"].append(
            (data, time_now, task_id)
        )
    else:
        current_app.config["INFERENCE_STEP_QUEUE"].append((data, time_now, task_id))
    return jsonify({"task_id": task_id}), 201


# POST: Forward pass on input data
@current_app.route("/direct_forward", methods=["POST"])
def direct_forward() -> tuple[Response, int]:
    """
    When POST request is made to /direct_forward:
    Return the direct model output. If the input is invalid, return an error.

    Returns:
        tuple[Response, int]: Tuple of the response and the status code.
    """
    data = request.json
    input_string = data.get("input_string", None)
    safe_logits = data.get("safe_logits", False)
    num_samples = data.get("num_samples", 1)
    temperature = data.get("temperature", 0.7)
    max_input_length = data.get("max_input_length", 1024)
    max_output_length = data.get("max_output_length", max_input_length + 256)
    decoding_strategy = data.get("decoding_strategy", "temperature")
    stop_strings = data.get("stop_strings", ["<|eot_id|>", "<|eois_id|>"])
    skip_special_tokens = data.get("skip_special_tokens", True)

    if not input_string:
        return jsonify({"error": "Invalid input"}), 400

    if safe_logits:
        return (
            jsonify({"error": "safe_logits is not implemented for direct_forward"}),
            501,
        )

    if decoding_strategy not in ["temperature", "diverse_beam_search", "beam_search"]:
        return (
            jsonify(
                {
                    "error": "Invalid decoding strategy. Supported strategies are 'temperature', 'diverse_beam_search', 'beam_search'."
                }
            ),
            400,
        )

    generated_texts, _, stop_strings_found = current_app.config[
        "MODEL_MANAGER"
    ].forward_prompt(
        input_prompt=[input_string],
        max_input_length=max_input_length,
        max_output_length=max_output_length,
        num_samples=num_samples,
        output_logits=safe_logits,
        temperature=temperature,
        stop_strings=stop_strings,
        decoding_strategy=decoding_strategy,
        skip_special_tokens=skip_special_tokens,
    )

    result_dict = {"result": generated_texts, "stop_strings": stop_strings_found}
    return jsonify(result_dict), 200


def start_TRAINING_THREAD(
    samples: list[dict], num_epochs: int, lr: float, app_context: AppContext
) -> None:
    """
    Function to start the background training process.

    The dictionaries in samples should contain at least the keys "GAE" and "messages".

    Args:
        samples (list[dict]): Samples to train on.
        num_epochs (int): Number of epochs to train.
        lr (float): Learning rate.
        app_context (AppContext): Application context.
    """
    app_context.push()  # Need this to have the same context as the main app
    current_app.config["MODEL_MANAGER"].ppo_train_on_dataset(samples, num_epochs, lr)
    current_app.config["ALLOW_TRAINING"] = True
    print("Training finished", flush=True)


# POST: Add multiple data samples or a dataset for fine-tuning
@current_app.route("/train_from_dataset", methods=["POST"])
def train_from_dataset() -> tuple[Response, int]:
    """
    When POST request is made to /train_from_dataset:
    Train on given dataset or sample. A sample is defined as (question, context, current_node,
    value). It is also possible to provide a path on the server from where the dataset should be
    loaded.

    Returns:
        tuple[Response, int]: Tuple of the response and the status code.
    """
    if not current_app.config["ALLOW_TRAINING"]:
        return jsonify({"error": "Training is already in progress"}), 403
    current_app.config["ALLOW_TRAINING"] = False
    current_app.config["STOP_TRAINING_FLAG"] = False

    data = request.json
    dataset = data.get("dataset", None)
    dataset_path = data.get("path", None)
    dataset_loader_function_name = data.get("dataset_loader_class", None)
    num_epochs = data.get("num_epochs", 1)
    percent_of_data = data.get("percent_of_data", 1.0)
    save_samples = data.get("save_samples", False)
    lr = data.get("lr", 1e-5)

    print(
        f"Samples: {dataset}, Path: {dataset_path}, Loader: {dataset_loader_function_name}, Epochs: {num_epochs}",
        flush=True,
    )

    if dataset_path is not None:
        return jsonify({"error": "Path is not implemented yet"}), 501
    if percent_of_data != 1.0:
        return jsonify({"error": "percent_of_data is not implemented yet"}), 501

    if dataset_path is not None and not dataset_loader_function_name:
        return jsonify({"error": "Please provide the dataset_loader_class."}), 400

    if dataset is None and dataset_path is None:
        return (
            jsonify(
                {
                    "error": "Invalid input, no data is provided. Provide either samples or path."
                }
            ),
            400,
        )

    if dataset is not None and dataset_path is not None:
        return (
            jsonify(
                {
                    "error": "Both a sample and a path are given. Please provide either samples or a path."
                }
            ),
            400,
        )

    if dataset is not None and save_samples:
        # Store the samples intermediately by creating a file in the directory $SCRATCH/datasets.
        # Check if there are any files labelled as Policy_training_xxx.json and create a new one
        # with the next number and store the samples in the file.
        datasets = os.listdir(f'{os.environ["SCRATCH"]}/datasets')
        datasets = [d for d in datasets if d.startswith("Policy_training")]
        if len(datasets) == 0:
            dataset_name = "Policy_training_0.json"
        else:
            dataset_name = f'Policy_training_{max([int(d.split("_")[-1]) for d in datasets]) + 1}.json'
        dataset_path = f'{os.environ["SCRATCH"]}/datasets/{dataset_name}'
        with open(f'{os.environ["SCRATCH"]}/datasets/{dataset_name}', "w") as f:
            f.write(dataset)

    # Set the number of CPUs to all but one for the training process
    num_cores = os.cpu_count() - 1  # Leave 1 core for Flask
    if num_cores < 1:
        num_cores = 1  # In case the system only has one core

    # Limit the number of CPU cores for the training process
    os.environ["OMP_NUM_THREADS"] = str(
        num_cores
    )  # Set OpenMP thread count for libraries like NumPy, TensorFlow

    current_app.config["TRAINING_THREAD"] = Thread(
        target=start_TRAINING_THREAD,
        args=(dataset, num_epochs, lr, current_app.app_context()),
    )
    current_app.config["TRAINING_THREAD"].start()

    return (
        jsonify(
            {
                "message": f"Training started with {num_cores} cores and {os.environ['CUDA_VISIBLE_DEVICES']} GPUs"
            }
        ),
        201,
    )


# Route to stop training
@current_app.route("/stop_training", methods=["POST"])
def stop_training() -> tuple[Response, int]:
    """
    When POST request is made to /stop_training:
    Stop training the model.

    Returns:
        tuple[Response, int]: Tuple of the response and the status code.
    """

    current_app.config["STOP_TRAINING_FLAG"] = True

    print("Training should stop now", flush=True)
    print("stop_training_flag: ", current_app.config["STOP_TRAINING_FLAG"], flush=True)

    # Wait for the training process to finish (if necessary)
    print("Training thread: ", current_app.config["TRAINING_THREAD"], flush=True)
    if current_app.config["TRAINING_THREAD"] is not None:
        print("Waiting for the training process to finish", flush=True)
        current_app.config["TRAINING_THREAD"].join()

    current_app.config["ALLOW_TRAINING"] = True
    current_app.config["STOP_TRAINING_FLAG"] = False

    return jsonify({"message": "Training stopped and model saved"}), 200
