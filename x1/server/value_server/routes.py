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


def check_list_of_dicts(input_message: list[dict]) -> bool:
    """
    Check if the input_message is a list of dictionaries with 'role' and 'content' fields.

    Args:
        input_message (list[dict]): List of dictionaries.
    Returns:
        bool: True if the input_message follows the desired scheme, False otherwise.
    """
    if not isinstance(input_message, list):
        return False
    for message in input_message:
        if not isinstance(message, dict):
            return False
        if "role" not in message or "content" not in message:
            return False
    return True


# POST: Forward pass on input data
@current_app.route("/forward", methods=["POST"])
def forward() -> tuple[Response, int]:
    """
    When POST request is made to /forward:
    Run inference on the payload's 'messages' field and return result.
    If the input is invalid, return an error.

    Returns:
        tuple[Response, int]: Tuple of the response and the status code.
    """
    data = request.json
    input_messages = data.get("messages", None)

    if not input_messages:
        return jsonify({"error": "Invalid input"}), 400

    # If the input message is a list of of lists,
    if not isinstance(input_messages, list):
        return jsonify({"error": "Invalid input, input messages is not a list."}), 400
    if all(isinstance(i, list) for i in input_messages):
        for i in input_messages:
            if not check_list_of_dicts(i):
                return (
                    jsonify(
                        {
                            "error": "Invalid input, input messages is not a list of dictionaries with keys 'role' and 'content'."
                        }
                    ),
                    400,
                )
        result = current_app.config["MODEL_MANAGER"].evaluate_reasoning_steps(
            messages=input_messages
        )
    else:
        if not check_list_of_dicts(input_messages):
            return (
                jsonify(
                    {
                        "error": "Invalid input, input messages is not a list of dictionaries with keys 'role' and 'content'."
                    }
                ),
                400,
            )
        result = current_app.config["MODEL_MANAGER"].evaluate_reasoning_steps(
            messages=[input_messages]
        )

    return jsonify({"result": result}), 200


# Function to start the background training process
def start_TRAINING_THREAD(
    samples: list[dict], num_epochs: int, lr: float, app_context: AppContext
) -> None:
    """
    Function to start the background training process.

    The dictionaries in samples should contain at least the keys "value_target" and "messages".

    Args:
        samples (list[dict]): Samples to train on.
        num_epochs (int): Number of epochs to train.
        lr (float): Learning rate.
        app_context (AppContext): Application context.
    """
    app_context.push()  # Need this to have the same context as the main app
    current_app.config["MODEL_MANAGER"].mse_train_on_dataset(samples, num_epochs, lr)
    current_app.config["ALLOW_TRAINING"] = True
    print("Training finished", flush=True)


# POST: Add multiple data samples or dataset for fine-tuning
@current_app.route("/train_from_dataset", methods=["POST"])
def train_from_dataset() -> tuple[Response, int]:
    """
    When POST request is made to /train_from_dataset:
    Train on a given dataset or sample. A sample is defined as (question, context, current_node,
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
        return jsonify({"error": "Path is not implemented yet."}), 501
    if percent_of_data != 1.0:
        return jsonify({"error": "percent_of_data is not implemented yet"}), 501

    if dataset_path is not None and not dataset_loader_function_name:
        return jsonify({"error": "Please provide the dataset_loader_class."}), 400

    if dataset is None and dataset_path is None:
        return (
            jsonify(
                {
                    "error": "Invalid input, no data is provided. Provide either samples or a path."
                }
            ),
            400,
        )

    if dataset is not None and dataset_path is not None:
        return (
            jsonify(
                {
                    "error": "Both samples and a path is given. Please provide either samples or a path."
                }
            ),
            400,
        )

    if dataset is not None and save_samples:
        # Store the samples intermediately by creating a file in the directory $SCRATCH/datasets.
        # Check if there are any files labelled as Value_training_xxx.json and create a new one
        # with the next number and store the samples in the file.
        datasets = os.listdir(f'{os.environ["SCRATCH"]}/datasets')
        datasets = [d for d in datasets if d.startswith("Value_training")]
        if len(datasets) == 0:
            dataset_name = "Value_training_0.json"
        else:
            dataset_name = f'Value_training_{max([int(d.split("_")[-1]) for d in datasets]) + 1}.json'
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
                "message": f"Training started with {num_cores} cores and {os.environ['CUDA_VISIBLE_DEVICES']} GPUs."
            }
        ),
        201,
    )


# Route to stop training
@current_app.route("/stop_training", methods=["POST"])
def stop_training() -> tuple[Response, int]:
    """
    When POST request is made to /stop_training:
    Stop the training process.

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
