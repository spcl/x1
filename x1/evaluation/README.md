# Reasoning Step Evaluation

The evaluation component handles the evaluation of reasoning steps.

## Description

The AbstractStepEvaluation class focuses on the evaluation of reasoning steps and provides two methods: the evaluation of a single reasoning step and of multiple reasoning steps.
We provide two implementations for the AbstractStepEvaluation class: a value model and evaluation through prompting.
The definition of the AbstractStepEvaluation class can be found in the file [abstract_step_evaluation.py](abstract_step_evaluation.py).

## Value Model

The ValueModel class uses a value model, which is hosted within the [server](../server) environment, to evaluate reasoning steps.
In addition to the two methods from the AbstractStepEvaluation class, the ValueModel class also implements an `__init__` method to provide a configuration for the value model.
In particular, the key `value_api_url` should be set to provide an URL to where the value model can be accessed.
The `evaluate_step` method queries the value model for a value of the provided reasoning step based on the reasoning structure given as the context, where the original task description is stored as the first item of the `contextual_memory` and whether the current reasoning step is final, i.e. a terminal node.
`evaluate_steps` works similarly, but takes lists for the respective arguments.
The value model is implemented in the file [value_model.py](value_model.py).

The ValueModel class does not provide functionality to determine the terminality of a reasoning step, since the terminality of the step is based on the `eos` token prediction by the policy model.

## Prompted Step Evaluation

The PromptedStepEvaluation class is based on prompting a Llama 3.1 70B model to evaluate reasoning steps.
The Llama model is accessed through the [Ollama](https://github.com/ollama/ollama) API and should be hosted on an Ollama server.
Similar to the ValueModel class, the PromptedStepEvaluation class also implements an `__init__` method to provide a configuration for accessing the Llama model.
The key `api_url` should be set to provide an URL to where the Llama model can be accessed.
During the prompting of the model for the evaluation of a reasoning step, the model generates a score between -100 (poor quality) and 100 (high quality).
Again the original task description is stored as the first item of the `contextual_memory` list and subsequent items of that list contain the previous reasoning structure.
The following prompt template is used for scoring a reasoning step:
```
**Task**: {contextual_memory[0]}

**Current Reasoning Step**: {step}

**Previous Reasoning Steps**: {contextual_memory[1:]}

**Instructions**:
You are given a reasoning step as a next step in a reasoning process to solve the given **Task**. Evaluate the **Current Reasoning Step** by considering its overall quality in the context of the problem and previous reasoning steps.
Take into account factors such as relevance to the problem, contribution to progress, and logical consistency.

- Assign an **Overall Score** between **-100 and 100**, where:
- **-100**: Poor quality.
- **0**: Moderate quality.
- **100**: High quality.

**Rules:**
- If the reasoning step contains errors, then it should be scored very low.
- If the reasoning step provides the correct answer to the task, then it should be scored as 100.
- The higher the contribution of the reasoning step to the correct answer, the higher it should be scored.

Provide a brief **justification** for the assigned score.

**Output**:
<analysis>your_reflective_and_strict_analysis</analysis>
<score>your_score_as_integer</score>
```

The PromptedStepEvaluation class also implements another method (`is_solved`) to determine the terminality of the reasoning step, i.e. whether the reasoning step solves the task based on the question and the previous reasoning steps.
The following prompt template is used for the terminality decision of a reasoning step:
```
Evaluate whether the reasoning step is the **terminal, complete solution** to the problem described in the **Task**.
A reasoning step is terminal if it contains the actual answer to the problem and is not only describing the process to solve the task.
For example, if it is a mathematical problem asking to find the value of x, it must contain the actual number associated with x. If it describes a process, the reasoning step must arrive at a conclusion that directly answers the question.

Output yes if the reasoning step is terminal or no if not inside the <is solved> html tag. Output nothing else.
If the possible solution describes only part of the process or leaves the final step unsolved, mark it as "no". Only mark it as "yes" if the reasoning step provides a complete, conclusive answer to the original question.

**Task**: {contextual_memory[0]}

**Step to evaluate**: {step}

**Previous reasoning steps**: {contextual_memory[1:]}

Output example:
<is solved>yes</is solved>
```
The implementation of the prompted evaluation can be found in the file [prompted_step_evaluation.py](prompted_step_evaluation.py).


## Ollama Setup

The following instructions provide an overview of setting up a model within the Ollama environment.

1. Install according to the [instructions](https://github.com/ollama/ollama/blob/main/docs/linux.md) in the Ollama repository.
2. Start the Ollama server with `ollama serve` and verify that the server is running by using `curl http://localhost:11434`.
3. Download the model by executing for example `ollama pull llama3.1:70b`.
4. If the server is running on a custom port or or a non-default model is being used, the respective parameters for the initialization of `PromptedStepExpansion` and `PromptedStepEvaluation` should be adjusted.
