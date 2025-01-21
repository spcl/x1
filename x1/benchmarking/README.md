# Benchmarking

The benchmarking component enables the efficient benchmarking of Reasoning Language Models by either using existing datasets or datasets based on the reasoning domains from the x1 framework.


## Description

There are two main classes, the `Generator` and the `Extractor`, as well as code for the actual benchmarking in this component.
Additionally a verifier for the [MATH](https://github.com/hendrycks/math) dataset is provided.


### Generator

The `Generator` generates a response for a given task either by directly querying a model or by using a reasoning strategy.
The interface of the AbstractGenerator class can be found in the file [abstract_generator.py](abstract_generator.py).

x1 provides four implementations for the AbstractGenerator class.
The StandardGenerator class sends a prompt to a model and returns the unmodified response, whereas the BoxedGenerator class takes the initial response and prompts the model a second time to include the solution inside a `\boxed{}` statement.
The MCTSGenerator class uses a Monte Carlo Tree Search (MCTS) to generate a response for a given task.
If the supplied extractor is able to to extract the solution from the response, just that solution will be returned, otherwise the solution is provided inside a `\boxed{}` statement.
Alternatively the MCTSGeneratorWithSimulations class can be chosen, where MCTS can be used with simulations to generate the final solution.
These four implementations are located in the file [generator.py](generator.py).


### Extractor

The `Extractor` extracts a solution from a response.
The interface of the AbstractExtractor class can be found in the file [abstract_extractor.py](abstract_extractor.py).

In addition to a DummyExtractor class, that returns the unmodified response as a solution, x1 also provides an extractor for the MATH dataset, which returns solutions found inside `\boxed{}` statements, after "The answer is" as well as after "####".
A portion of the MATHExtractor class stems from the [GSM8K](https://github.com/openai/grade-school-math) benchmark.
The code of these implementations is located in the file [extractor.py](extractor.py).


## Benchmarking Code

A reference implementation for a benchmark code can be found in the file [benchmark.py](benchmark.py).
The benchmark first generates reponses for a dataset of tasks with a subsequent extraction and verification of the actual answers from the model responses and stores the results (unmodified responses, extracted solutions and verification) in an output file.
Additionally the benchmarks computes the aggregated accuracy for the model on the given dataset.
The benchmark uses the [AbstractGenerator](#generator), the [AbstractExtractor](#extractor) and [AbstractVerifier](../synthetic_generation#verifier) classes.

x1 also provides a specific verifier for the MATH dataset in the file [math_verifier.py](math_verifier.py), which implements the [AbstractVerifier](../synthetic_generation#verifier) API.


## Example

An example on how to use the benchmarking component is provided in the [examples](../../examples) directory, specifically in the file [example_benchmark.py](../../examples/example_benchmark.py).
