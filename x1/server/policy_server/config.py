# Copyright (c) 2025 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# main author: Eric Schreiber
#
# contributions: Afonso Catarino

import os
from collections import deque
from datetime import datetime

from x1.server.policy_server.policy_model import PolicyModel


class BaseConfig:
    """
    Base configuration for the server.
    """

    def __init__(self, GPU: int) -> None:
        """
        Initialize the configuration.

        Args:
            GPU (int): GPU to use.
        """
        self.GPU = GPU

        self.STOP_TRAINING_FLAG = False
        self.ALLOW_TRAINING = True
        self.OUTPUT_FOLDER = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "outputs"
        )
        # Create a directory with the current date and time including microseconds
        current_time = datetime.now()
        self.OUTPUT_FOLDER = os.path.join(
            self.OUTPUT_FOLDER,
            f"experiment_{current_time.year}_{current_time.month}_{current_time.day}_{current_time.hour}_{current_time.minute}_{current_time.second}_{current_time.microsecond}",
        )
        os.makedirs(self.OUTPUT_FOLDER)

        self.FORWARD_PATH = os.path.join(self.OUTPUT_FOLDER, "forward_pass")
        self.LOGITS_PATH = os.path.join(self.FORWARD_PATH, "logits")
        self.FORWARD_PASS_FILE_PATH = os.path.join(
            self.FORWARD_PATH, "generated_sentences.csv"
        )
        self.PROMPT_FILE_PATH = os.path.join(self.FORWARD_PATH, "prompts.csv")
        os.makedirs(self.FORWARD_PATH)
        os.makedirs(self.LOGITS_PATH)
        with open(self.FORWARD_PASS_FILE_PATH, "w") as f:
            f.write("")
        with open(self.PROMPT_FILE_PATH, "w") as f:
            f.write("")

        self.INPUT_MODEL_PATH = os.path.join(self.OUTPUT_FOLDER, "models")
        self.BATCH_SIZE = 64  # 64 with GEMMA 2B, 32 with LLama 3.1 8B
        self.INFERENCE_BATCH_SIZE = 32
        self.BATCH_SIZE_TRAIN = 4
        self.MODEL_MANAGER = PolicyModel(
            base_model_path="<Set path to base model.>",
            LoRa_model_path="<Set path to LoRa model.>",
            save_path=os.path.join(self.OUTPUT_FOLDER, "models"),
            lora_training=True,
            device=self.GPU,
        )

        self.TRAINING_THREAD = None
        self.INFERENCE_THREAD = None
        self.INFERENCE_FINISHED_DICT = {}
        # Two queues because reasoning steps and simulations take a different amount of time, so
        # they would slow each other down.
        self.INFERENCE_STEP_QUEUE = deque()
        self.INFERENCE_SIMULATION_QUEUE = deque()
