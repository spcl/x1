# Example Codes

This directory provides a number of short example codes to illustrate certain aspects of the x1 framework.

## Reasoning Task Generation and Verification

[example_synthetic_generation.py](example_synthetic_generation.py) generates a single task for three reasoning domains (SAT, Knapsack, PrOntoQA) and tries to verify a dummy model output with the verifier of the respective reasoning domain.

[example_synthetic_dataset_generation.py](example_synthetic_dataset_generation.py) illustrates how to generate a dataset of SAT tasks with different levels of difficulty and stores these tasks in a JSON file with the following format:
```json
{
    "version": 1.0,
    "data": [
        {
            "type": "SAT",
            "level": 1,
            "problem": "Consider the following propositional logic formula\n\n(NOT x_2 OR x_1 OR NOT x_5) AND (NOT x_4 OR x_2 OR NOT x_1) AND (NOT x_5 OR x_1 OR NOT x_3) AND (NOT x_1 OR x_3 OR NOT x_5) AND (x_4 OR x_2 OR x_5) \n\nProvide a value for each variable (x_i = True / False), such that the formula is satisfied.",
            "unique_solution": false,
            "solution": "(NOT x_2 OR x_1 OR NOT x_5) AND (NOT x_4 OR x_2 OR NOT x_1) AND (NOT x_5 OR x_1 OR NOT x_3) AND (NOT x_1 OR x_3 OR NOT x_5) AND (x_4 OR x_2 OR x_5) ",
            "id": 0
        },
        ...
    ]
}
```

## Benchmarking

[example_benchmark.py](example_benchmark.py) presents a code that runs a benchmark on a number of SAT tasks in a dataset, where a locally hosted model generates solutions to these tasks, which are then verified.
The accuracy on the dataset is returned at the end and the results are stored in `SAT_result.json`.
We provide a SAT dataset with a hundred tasks in the file `SAT_dataset.json` in the format described above, which can be used as an input.
The output file uses a similar format as the input file with a few added name-value pairs:

```json
{
    "version": 1.0,
    "data": [
        {
            "type": "SAT",
            "level": 1,
            "problem": "Consider the following propositional logic formula\n\n(NOT x_2 OR x_1 OR NOT x_5) AND (NOT x_4 OR x_2 OR NOT x_1) AND (NOT x_5 OR x_1 OR NOT x_3) AND (NOT x_1 OR x_3 OR NOT x_5) AND (x_4 OR x_2 OR x_5) \n\nProvide a value for each variable (x_i = True / False), such that the formula is satisfied.",
            "unique_solution": false,
            "solution": "(NOT x_2 OR x_1 OR NOT x_5) AND (NOT x_4 OR x_2 OR NOT x_1) AND (NOT x_5 OR x_1 OR NOT x_3) AND (NOT x_1 OR x_3 OR NOT x_5) AND (x_4 OR x_2 OR x_5) ",
            "id": 0,
            "output": "Hmm... I don't know the answer to this one.",
            "extracted_output": "Hmm... I don't know the answer to this one.",
            "extracted_answer": "(NOT x_2 OR x_1 OR NOT x_5) AND (NOT x_4 OR x_2 OR NOT x_1) AND (NOT x_5 OR x_1 OR NOT x_3) AND (NOT x_1 OR x_3 OR NOT x_5) AND (x_4 OR x_2 OR x_5)",
            "correct": False
        },
        ...
    ]
}
```

You have to set the path to the model, so that it can be loaded locally.

## Monte Carlo Tree Search (MCTS)

[example_mcts.py](example_mcts.py) uses Monte Carlo tree search (MCTS) with a policy model, but no value model, to solve a single task from the MATH dataset. Please run the [server](../x1/server) locally to use this example.
[example_prompted_mcts.py](example_prompted_mcts.py) runs MCTS on the same task, but uses prompted step generation and evaluation instead. Please run an Ollama server locally to use this example.
