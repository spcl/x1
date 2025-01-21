# Copyright (c) 2025 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# main author: Afonso Catarino
#
# contributions:
# Robert Gerstenberger
# Hannes Eberhard

from x1.synthetic_generation import (
    EinsteinGenerator,
    EinsteinVerifier,
    KnapsackGenerator,
    KnapsackVerifier,
    PrOntoQAGenerator,
    PrOntoQAVerifier,
    SATGenerator,
    SATVerifier,
)

# Generate model output
dummy_output = "Hmm... I don't know the answer to this one."

"""
SAT
"""
# Generate a SAT instance
sat_generator = SATGenerator()
task = sat_generator.generate(difficulty=0.5)

# Verify model output
sat_verifier = SATVerifier()
print("SAT\n===")
print(
    f"task:\n{task['problem']}\nmodel output:\n{dummy_output}\nverifier output:\n{sat_verifier.verify(task, dummy_output)}\n\n"
)


"""
Knapsack
"""
# Generate a knapsack instance
knapsack_generator = KnapsackGenerator()
task = knapsack_generator.generate(difficulty=0.5)

# Verify model output
knapsack_verifier = KnapsackVerifier()
print("knapsack\n========")
print(
    f"task:\n{task['problem']}\nmodel output:\n{dummy_output}\nverifier output:\n{knapsack_verifier.verify(task, dummy_output)}\n\n"
)


"""
Einstein
"""
# Generate an Einstein riddle instance
einstein_generator = EinsteinGenerator()
task = einstein_generator.generate(difficulty=0.5)

# Verify model output
einstein_verifier = EinsteinVerifier()
print("Einstein\n========")
print(
    f"task:\n{task['problem']}\nmodel output:\n{dummy_output}\nverifier output:\n{einstein_verifier.verify(task, dummy_output)}\n\n"
)


"""
PrOntoQA
"""
# Generate a PrOntoQA instance
prontoqa_generator = PrOntoQAGenerator()
task = prontoqa_generator.generate(difficulty=0.5)

# Verify model output
prontoqa_verifier = PrOntoQAVerifier()
print("PrOntoQA\n========")
print(
    f"task:\n{task['problem']}\nmodel output:\n{dummy_output}\nverifier output:\n{prontoqa_verifier.verify(task, dummy_output)}"
)
