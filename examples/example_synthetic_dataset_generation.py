# Copyright (c) 2025 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# main author: Afonso Catarino

import json

from x1.synthetic_generation import SATGenerator

sat_generator = SATGenerator(seed=1)

num_samples = 100
dataset = []
for i in range(num_samples):
    task = sat_generator.generate(difficulty=i / num_samples)
    dataset.append(task)

with open("SAT_dataset.json", "w") as file:
    json.dump({"version": 1, "data": dataset}, file, indent=4)
