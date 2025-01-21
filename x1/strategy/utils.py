# Copyright (c) 2025 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# main author: Eric Schreiber
#
# contributions: Afonso Catarino

import requests
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_delay,
    wait_random_exponential,
)
from torchtyping import TensorType


class MinMaxStats:
    """
    Keep a running minimum and maximum of a set of values.
    """

    def __init__(self, minimum: float, maximum: float) -> None:
        """
        Initialize the minimum and maximum values.

        Args:
            minimum (float): Minimum value.
            maximum (float): Maximum value.
        """
        self.maximum = maximum
        self.minimum = minimum

    def update(self, value: float) -> None:
        """
        Update minimum and maximum values given a new value.
        If value is zero, no update is performed.

        Args:
            value (float): New value to update.
        """
        if value == 0:  # Do not update when value has the value zero
            return
        self.maximum = max(self.maximum, value)
        self.minimum = min(self.minimum, value)

    def normalize(self, value: float | TensorType[float]) -> float | TensorType[float]:
        """
        Normalize the value between the minimum and maximum values

        Args:
            value (float | TensorType[float]): Value(s) to normalize.
        Returns:
            float | TensorType[float]: Normalized value(s).
        """
        if self.maximum > self.minimum:
            # We normalize only when we have set the maximum and minimum values.
            return (value - self.minimum) / (self.maximum - self.minimum)
        return value


def send_request(payload: dict, api_url: str) -> dict:
    """
    Send a request to the server with the payload and return the response.

    Args:
        payload (dict): Payload to send to the server.
        api_url (str): Url of the server.
    Returns:
        dict: Response from the server.
    """
    response = requests.post(api_url, json=payload)
    try:
        response.raise_for_status()
    except requests.exceptions.HTTPError as e:
        print(
            f"Error: {e} with response: {response.json()} for payload: {payload} and api_url: {api_url}.",
            flush=True,
        )
        raise e
    response_json = response.json()
    return response_json


# retry with exponential backoff, stop after 30 attempts, wait between 1 and 20 seconds
@retry(
    retry=retry_if_exception_type(requests.exceptions.HTTPError),
    stop=stop_after_delay(500),
    wait=wait_random_exponential(multiplier=1, max=16),
)
def get_server_response(task_id, api_url: str) -> dict:
    """
    Get the response from the model hosted by the server given a task id.
    Retry with exponential backoff, stop after 30 attempts, wait between 1 and 20 seconds.

    Args:
        task_id: Id of the task.
        api_url (str): Url of the server.
    Returns:
        dict: Response from the server.
    """
    response = requests.get(f"{api_url}/{task_id}")
    response.raise_for_status()
    response_json = response.json()
    return response_json


def send_queued_server_request(payload: dict, api_url: str) -> dict:
    """
    Send a request to the server with the payload and return the model response to the task.

    Args:
        payload (dict): Payload to send to the server.
        api_url (str): Url of the server.
    Returns:
        dict: Response from the server.
    """
    if api_url[-1] == "/":
        api_url = api_url[:-1]
    # Send the request to the server, get a task id
    response = send_request(payload, api_url)
    task_id = response["task_id"]
    return get_server_response(task_id, api_url)
