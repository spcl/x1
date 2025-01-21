# Copyright (c) 2025 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# main author: Eric Schreiber
#
# contributions:
# Afonso Catarino
# Hannes Eberhard

"""
Example code to run a benchmark on SAT tasks with a locally hosted generative model.

Please set the path to the model.
"""

import json
from os.path import dirname, realpath

from transformers import AutoModelForCausalLM, AutoTokenizer

from datasets import Dataset
from x1.benchmarking import DummyExtractor, StandardGenerator
from x1.benchmarking.benchmark import compute_benchmark
from x1.synthetic_generation import SATVerifier

parent = dirname(realpath(__file__))

# Load dataset
with open(f"{parent}/SAT_dataset.json", "r") as f:
    dataset = Dataset.from_list(json.load(f)["data"])

# Load the model
model_name = "<path_to_model>"

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = "<|finetune_right_pad_id|>"  # https://discuss.huggingface.co/t/how-to-set-the-pad-token-for-meta-llama-llama-3-models/103418/5

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    low_cpu_mem_usage=True,
    device_map="auto",
)

# Initialize the generator, extractor and verifier
generator = StandardGenerator(model, tokenizer, max_new_tokens=4096)
extractor = DummyExtractor()
verifier = SATVerifier()

# Evaluate the model
result = compute_benchmark(
    dataset=dataset,
    generator=generator,
    extractor=extractor,
    verifier=verifier,
    output_file_path=f"{parent}/SAT_result.json",
)

print(f"Reached an accuracy of {result} for the SAT dataset.")
