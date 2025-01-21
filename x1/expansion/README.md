# Reasoning Step Expansion

The expansion component handles the generation of new reasoning steps.

## Description

The AbstractStepExpansion class provides two methods: the generation of a specific number of reasoning steps based on the current contextual memory as well as the simulation of the reasoning process until the end.
We provide two implementations for the AbstractStepExpansion class: a policy model and expansion through prompting.
The definition of the AbstractStepExpansion class can be found in the file [abstract_step_expansion.py](abstract_step_expansion.py).

### Policy Model

The PolicyModel class uses a policy model, which is hosted within the [server](../server) environment, to generate new reasoning steps.
In addition to the two methods from the AbstractStepExpansion class, the PolicyModel class also implements an `__init__` method to provide a configuration for the policy model.
In particular, the key `policy_api_url` should be set to provide an URL to where the policy model can be accessed.
The `generate_steps` method generates `num_steps` independent reasoning steps based on the reasoning structure given as the context, where the original task description is stored as the first item of the `contextual_memory` list and subsequent items of that list contain the previous reasoning structure.
As decoding strategy, the policy model uses [Diverse Beam Search](https://ojs.aaai.org/index.php/AAAI/article/view/12340).
The policy model is implemented in the file [policy_model.py](policy_model.py).

The terminality of a reasoning step is determined by the `eos` token prediction of the policy model.

### Prompted Step Expansion

The PromptedStepExpansion class is based on prompting a Llama 3.1 70B model to generate new reasoning steps.
The Llama model is accessed through the [Ollama](https://github.com/ollama/ollama) API and should be hosted on an Ollama server.
Similar to the PolicyModel class, the PromptedStepExpansion class also implements an `__init__` method to provide a configuration for accessing the Llama model.
The key `api_url` should be set to provide an URL to where the Llama model can be accessed.
During the prompting of the model for the next reasoning steps, the model generates `num_steps` such steps within the same context.
Again the original task description is stored as the first item of the `contextual_memory` list and subsequent items of that list contain the previous reasoning structure.
The implementation of the prompted expansion can be found in the file [prompted_step_expansion.py](prompted_step_expansion.py).
The following prompt template is used for evaluating a reasoning step:
```
<question>{contextual_memory[0]}</question>
<step>{step}</step>
<past_chain>{contextual_memory[1:]}</past_chain>

You are given a question and a chain of steps. The chain of steps contains the past chain and the reasoning step that is the starting point for your task.
Your task is to simulate the reasoning process after the step in the reasoning chain to solve the question.
- **Do not ignore the given reasoning chain or reasoning step** by solving the question directly. Your simulation is used to evaluate the reasoning step so its capabilities are tested through your simulation.
- **Follow logically** from the current state, reasoning step-by-step until you reach a terminal conclusion.
- Ensure that every intermediate step is logically consistent and that the reasoning is **free from errors**.
- The goal is to explore whether you can logically progress to a final solution based on the current context.
- You need to extract the final result from your process that is requested in the question based on the given reasoning process.

Make sure to not introduce any mistakes in the reasoning process, and output a solution only if it logically follows from the provided chain of steps.
For the output, first provide the reasoning chain till the end in the <simulation> tags. Then output the final result always into <answer> tags in the last line. Do not use the <answer> tag anywhere than in the last line for the final answer of your simulation.

**Output format:**
<simulation>reasoning_chain_here</simulation>
<answer>final_answer_here</answer>
```


## Ollama Setup

The following instructions provide an overview of setting up a model within the Ollama environment.

1. Install according to the [instructions](https://github.com/ollama/ollama/blob/main/docs/linux.md) in the Ollama repository.
2. Start the Ollama server with `ollama serve` and verify that the server is running by using `curl http://localhost:11434`.
3. Download the model by executing for example `ollama pull llama3.1:70b`.
4. If the server is running on a custom port or or a non-default model is being used, the respective parameters for the initialization of `PromptedStepExpansion` and `PromptedStepEvaluation` should be adjusted.
