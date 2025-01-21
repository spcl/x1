# Synthetic Generation

An important component for the effective training of Reasoning Language Models is the data.
The synthetic generation component focuses on the generation of synthetic data and the evaluation of model outputs.


## Description

There are two main classes in this component: the `Generator` and the `Verifier`.


### Generator

The `Generator` generates data via the `generate` method, which takes a `difficulty` argument (a `float` in the range `[0, 1]`) and returns a task with its attributes within a `dict` in the following format:
```python
{
    "type": str, # task domain
    "level": int, # difficulty level in [1, 10]
    "problem": str, # task description
    "unique_solution": bool, # whether the task has a unique solution
    "solution": str, # solution to the task
}
```
The difficulty arguments influences the difficulty level of the generated task.
The API definition for the generator is located in the file [abstract_generator.py](abstract_generator.py).


### Verifier

The `Verifier` evaluates model outputs via the `verify` method, which results in an integer score.
This method takes as arguments a `task` (the task description as returned by the generate method) and an `output` (the output string of a language model) and returns a score for the model output with the following possible values:

- -1: The output is invalid.
- 0: The output is incorrect.
- 1: The output satisfies the constraints of the problem. It may lead to a correct or an incorrect solution.
- 2: The problem was solved.

The API definition for the verifier is located in the file [abstract_verifier.py](abstract_verifier.py).


## Domains

We provide reference implementations for three reasoning domains (SAT, knapsack and Einstein's riddle) as well as an adaption of the [PrOntoQA](https://github.com/asaparov/prontoqa) framework to our interface.


### Boolean Satisfiability (SAT)

The boolean satisfiability problem, which is NP-hard, is one of the most well-known problems in computer science and can be posed as follows: Given a propositional logic formula, find an assignment of the variables such that the formula is satisfied.
The implementation restricts the generated tasks to 3-SAT.
While there may be multiple solutions for a generated task, the task generator ensures that there is at least one such solution.
The implementation can be found in the file [sat.py](sat.py).

While the difficulty of a 3-SAT problem is hard to gauge, the task generator considers the number of clauses in the formula in relation to the number of variables ratio of that formula.
The higher the ratio, the least likely it is to be satisfiable.
As the task generator guarantees the satisfiability of any generated task this ratio can serve as a proxy for the difficulty of the problem.

As the SAT Generator produces not necessarily a unique solution, the 'solution' field of the output includes just the formula in order for the verifier to check a proposed solution from a model more easily against the formula.


#### Example Task

Consider the following propositional logic formula

(NOT x_2 OR x_1 OR x_5) AND (NOT x_4 OR x_5 OR NOT x_1) AND (NOT x_4 OR x_5 OR x_3) AND (NOT x_2 OR NOT x_1 OR NOT x_5) AND (NOT x_5 OR x_1 OR NOT x_2)

Provide a value for each variable (x_i = True / False), such that the formula is satisfied.


### Knapsack

Another NP-hard problem is knapsack.
The setting of the problem is simple: Given are a backpack with a certain weight limit and various items with associated weights and values, the objective is to maximize the value of the items in the backpack while respecting the weight limit.
The decision problem version of knapsack includes an additional value target.
The objective is then to decide, whether or not the value target can be reached while respecting the weight limit.
The implementation can be found in the file [knapsack.py](knapsack.py).

The difficulty is tuned using two parameters: the number of items and how close to the maximum value the value target lies.
Increasing the number of items increases the size of the search space, where as a smaller difference between the value target ant the maximum value makes it harder to distinguish whether or not the value target is reachable.


#### Example Task

Imagine you're packing for a trip. You are going to carry your backpack for a long time, so you don't want its weight to exceed 40kg. The backpack itself weighs 1kg. Each item you are considering to pack has a weight and a value. You want the items in your backpack to amount to the highest possible value while maintaining the weight of the backpack below the aforementioned threshold. You are considering the following items and can only pack one of each:

pen weight: 5 value: 19

map weight: 3 value: 9

book weight: 4 value: 16

True or False: There is a list of items whose value sum is higher than 61 and weight sum is lower than 39.


### Einstein's Riddle

The Einstein Riddles (or Zebra Puzzles) are a class of logic puzzles that are used to test deductive reasoning skills.
The goal is to use clues to deduce the relationships between a set of entities.
Traditionally, the puzzles involve a row of houses and the attributes of the people living in them.
An example of such a setup is shown below:

| Position    | 1         | 2      | 3         | 4           | 5         |
| ----------- | --------- | ------ | --------- | ----------- | --------- |
| Name        | Brian     | Robert | Stephanie | Amanda      | William   |
| Food        | Pasta     | Salad  | Sandwich  | Lasagna     | Nachos    |
| Drink       | Milkshake | Beer   | Smoothie  | Bloody Mary | Margarita |
| House Color | Magenta   | Cyan   | Black     | Green       | Red       |

The task requires to assign each attribute to the correct house.
The difficulty of the task is determined by the number of houses and attributes involved.

The implementation can be found in the file [einstein.py](einstein.py).


#### Example Task

There are 5 houses in a row (numbered 1 to 5).

Each house is occupied by a a different person and each house/person has an unique attribute from the following sets: \
Name: Amanda, Brian, Robert, Stephanie, William \
Food: Lasagna, Nachos, Pasta, Salad, Sandwich \
Drink: Beer, Bloody Mary, Margarita, Milkshake, Smoothie \
House Color: Black, Cyan, Green, Magenta, Red

You are given the following information:
1. Nachos (Food) is equal to Red (House Color)
2. Stephanie (Name) is directly to the left of Bloody Mary (Drink)
3. Margarita (Drink) is adjacent to Amanda (Name)
4. William (Name) is equal to Margarita (Drink)
5. Robert (Name) is equal to Salad (Food)
6. Bloody Mary (Drink) is equal to 4 (Position)
7. Black (House Color) is directly to the right of 2 (Position)
8. Smoothie (Drink) is adjacent to Lasagna (Food)
9. Lasagna (Food) is adjacent to Stephanie (Name)
10. Pasta (Food) is equal to Brian (Name)
11. Green (House Color) is directly to the right of 3 (Position)
12. Milkshake (Drink) is directly to the left of Salad (Food)
13. Brian (Name) is equal to 1 (Position)
14. Beer (Drink) is directly to the left of Black (House Color)
15. Cyan (House Color) is adjacent to 1 (Position)

Which house is associated with Sandwich (Food)? Format your answer as \boxed{x} where x is the number of the house.


### PrOntoQA

[PrOntoQA](https://github.com/asaparov/prontoqa) is a data generation pipeline created to evaluate the reasoning capabilities of earlier GPT models, including GPT-3 and InstructGPT. We recycle this pipeline, adapt it for our needs and include it in our framework.

PrOntoQA stands for **Pr**oof and **Onto**logy-Generated **Q**uestion-**A**nswering. The task setting is simple. Given are a set of axioms and a target statement. The objective is to use the axioms and the Modus Ponens rule to prove the target statement. Recall, that Modus Ponens is a rule of inference in formal logic. It states that, if $P$ is true and $P \implies Q$ is true, then Q must be true as well.

The difficulty of a problem is solely defined by the number of deduction steps necessary to prove a statement, i.e., the number of Modus Ponens rule applications. It is easy to see, that longer deduction chains lead to harder problems.

Please note in order to generate reproducible datasets, do not use more than one PrOntoQA generator as the seed for the random number generation is set globally for this reasoning domain in contrast to the others.


#### Example Task

Lorpuses are not sweet. Shumpuses are not bright. Lorpuses are jompuses. Each impus is bright. Lempuses are not feisty. Every shumpus is a lempus. Each shumpus is a zumpus. Max is a lorpus. Max is a shumpus.

True or false: Max is not bright.


### Adding New Reasoning Domains

A new reasoning domain for the synthetic generation component can be introduced by writing new Generator and Verifier classes that implement the API of the respective abstract classes defined in the files [abstract_generator.py](abstract_generator.py) and [abstract_verifier.py](abstract_verifier.py).


## Example

An example on how to use the synthetic generation component is provided in the [examples](../../examples) directory, specifically in the file [example_synthetic_generation.py](../../examples/example_synthetic_generation.py).
Additionally example datasets with 1,000 samples for all four reasoning domains are provided in the [datasets](../../datasets) directory.
