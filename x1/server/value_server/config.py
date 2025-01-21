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
from datetime import datetime

from x1.server.value_server.value_model import LLMBasedValueModel


class BaseConfig:
    """
    Base configuration for the server.
    """

    def __init__(self, GPU_ID):
        """
        Initialize the configuration.

        Args:
            GPU_ID (int): ID of the GPU to use.
        """
        self.GPU_ID = GPU_ID

        self.STOP_TRAINING_FLAG = False
        self.ALLOW_TRAINING = True

        self.OUTPUT_FOLDER = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "outputs"
        )
        current_time = datetime.now()
        self.OUTPUT_FOLDER = os.path.join(
            self.OUTPUT_FOLDER,
            f"experiment_{current_time.year}_{current_time.month}_{current_time.day}_{current_time.hour}_{current_time.minute}_{current_time.second}_{current_time.microsecond}",
        )
        os.makedirs(self.OUTPUT_FOLDER)
        # Create a new directory called "experiment_X" with X being the smallest integer that is
        # not already taken.
        i = 1
        while os.path.exists(
            os.path.join(self.OUTPUT_FOLDER, "experiment_{}".format(i))
        ):
            i += 1
        self.OUTPUT_FOLDER = os.path.join(self.OUTPUT_FOLDER, "experiment_{}".format(i))
        os.makedirs(self.OUTPUT_FOLDER)

        self.LOSS_PAD_VALUE = -100

        self.INPUT_MODEL_PATH = os.path.join(self.OUTPUT_FOLDER, "models")
        self.BATCH_SIZE_TRAIN = 8  # with RL training
        self.BATCH_SIZE_EVAL = 64  # 64 with LoRa, 32 without
        self.BATCH_SIZE = 32

        # Manually specify the training type or let it be defined by the last layer.
        self.TRAINING_TYPE = None

        self.MODEL_MANAGER = LLMBasedValueModel(
            base_model_path="<Set path to base model.>",
            LoRa_model_path="<Set path to LoRa model.>",
            save_path=os.path.join(self.OUTPUT_FOLDER, "models"),
            lora_training=True,
            device=self.GPU_ID,
            loss_padding_value=self.LOSS_PAD_VALUE,
        )

        self.TRAINING_THREAD = None
