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
from typing import Any

from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast


class AbstractGenerator(ABC):
    """
    Abstract base class that defines the interface for generating model responses.
    """

    def __init__(
        self,
        model: Any,
        tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
        max_new_tokens: int = 1024,
        skip_special_tokens: bool = True,
    ) -> None:
        """
        Initialize the AbstractGenerator instance with the configuration and model details.

        Args:
            model (Any): Model to be queried.
            tokenizer (PreTrainedTokenizer | PreTrainedTokenizerFast): Tokenizer to be used.
            max_new_tokens (int): Maximum number of tokens to generate. Defaults to 1024.
            skip_special_tokens (bool): Flag, which indicates whether to skip the generation of
                                        special tokens. Defaults to True.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        self.skip_special_tokens = skip_special_tokens

    @abstractmethod
    def generate_response(self, input_chat: list[dict]) -> str:
        """
        Generate a model response for a given prompt.

        Args:
            input_chat (list[dict]): Prompt.
        Returns:
            str: Model response.
        """
        pass
