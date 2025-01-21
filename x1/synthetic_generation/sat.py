# Copyright (c) 2025 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# main author: Afonso Catarino
#
# contributions: Hannes Eberhard

import re

from z3 import Bool, Or, Solver, sat

from .abstract_generator import AbstractGenerator
from .abstract_verifier import AbstractVerifier


def solver_to_str(solver: Solver) -> str:
    """
    Convert a Z3 solver to a string.

    Args:
        solver (Solver): Z3 solver.
    Returns:
        str: String representation of the formula.
    """
    string = ""
    for clause in solver.assertions():
        string += "("
        for lit in clause.children():
            if len(lit.children()) > 0:
                string += "NOT "
                string += str(lit.children()[0]) + " OR "
            else:
                string += str(lit) + " OR "
        string = string[:-4]
        string += ") AND "
    string = string[:-4]
    return string


def parse_output(txt: str) -> list:
    """
    Parse model output for variable assignments.

    Args:
        txt (str): Output of a language model.
    Returns:
        list: List of variable assignments.
    """
    pattern = r"x_?(\d+)[^.;,\(\)\n]*\b(true|false)\b(?!(?:true|false)(?:\b))"
    matches = re.findall(pattern, txt, re.IGNORECASE)
    return matches


def parse_formula(txt: str) -> list:
    """
    Parse formula from the task description.

    Args:
        txt (str): Task description.
    Returns:
        list: List of clauses.
    """
    clauses = txt.split(" AND ")
    formula = []
    for clause in clauses:
        clause = clause.replace("(", "")
        clause = clause.replace(")", "")
        literals = clause.split(" OR ")
        for i in range(len(literals)):
            if "NOT" in literals[i]:
                literals[i] = -int(literals[i].split("_")[-1])
            else:
                literals[i] = int(literals[i].split("_")[-1])
        formula.append(literals)
    return formula


class SATGenerator(AbstractGenerator):
    """
    The SATGenerator class generates tasks within the SAT domain.

    Inherits from the AbstractGenerator class and implements its abstract methods.
    """

    def __init__(self, seed: int | None = None) -> None:
        """
        Initialize the SATGenerator instance with a seed for the random number generation as well
        as other necessary data structures.

        Args:
            seed (int | None): Seed for the random number generation. Defaults to None.
        """
        super().__init__(seed)
        self.num_vars = 5  # default number of variables for the SAT clauses
        self.vars = [Bool(f"x_{var_ind}") for var_ind in range(1, self.num_vars + 1)]

    def generate(self, difficulty: float) -> dict:
        """
        Generate a task in the SAT domain.

        The difficulty determines the ratio of the number of clauses to the number of variables for
        the SAT formula, which serves a proxy for the hardness of the task.

        Output format:
        {
            "type": str, # task domain
            "level": int, # difficulty level in [1, 10]
            "problem": str, # task description
            "unique_solution": bool, # whether the task has a unique solution
            "solution": str, # solution to the task
        }

        Args:
            difficulty (float): Difficulty level in [0, 1].
        Returns:
            dict: Task description.
        """

        alpha = 1 + difficulty * 3.4
        num_clauses = int(self.num_vars * alpha)

        satisfiable = False
        num_tries = 0
        while not satisfiable:
            if num_tries >= 10:
                num_clauses -= 1
                num_tries = 0

            solver = self.generate_solver(self.vars, num_clauses)
            num_tries += 1

            if solver.check() == sat:
                satisfiable = True

        formula = solver_to_str(solver)

        problem = "Consider the following propositional logic formula\n\n"
        problem += f"{formula}\n\n"
        problem += "Provide a value for each variable (x_i = True / False), such that the formula is satisfied."

        task = {}
        task["type"] = "SAT"
        task["level"] = int(difficulty * 10) + 1
        task["problem"] = problem
        task["unique_solution"] = False
        task["solution"] = formula

        return task

    def generate_solver(self, vars: list, num_clauses: int) -> Solver:
        """
        Generate a random SAT formula.

        Args:
            vars (list): List of variables.
            num_clauses (int): Number of clauses in the formula.
        Returns:
            Solver: Z3 solver.
        """
        solver = Solver()
        for i in range(num_clauses):
            clause = self.random.sample(vars, k=3)
            for j in range(len(clause)):
                if self.random.random() < 0.5:
                    clause[j] = ~clause[j]
            solver.append(Or(*clause))
        return solver


class SATVerifier(AbstractVerifier):
    """
    The SATVerifier class verifies tasks and potential solutions within the SAT domain.

    Inherits from the AbstractVerifier class and implements its abstract methods.
    """

    def verify(self, task: dict, output: str) -> int:
        """
        Verify the output of a model for a given task.

        The output is scored with the following scheme:
        0: The output is incorrect.
        1: The output satisfies the constraints of the problem. It may lead to a correct or
           an incorrect solution.
        2: The problem was solved.

        If a variable is assigned a value multiple times, only the last variable assignment is
        taken.

        Args:
            task (dict): Task description as returned by the generator.
            output (str): Model output.
        Returns:
            int: Score for the output.
        """

        parse = parse_output(output)
        formula = parse_formula(task["solution"])

        candidates = dict()
        for var, val in parse:
            if val.lower() in ["true"]:
                candidates[int(var)] = 1
            elif val.lower() in ["false"]:
                candidates[int(var)] = -1
            else:
                return -1

        if len(candidates) == 0:
            return 0

        for candidate in candidates:
            if candidate not in list(map(abs, sum(formula, []))):
                return -1

        number_sat_clauses = 0
        for clause in formula:
            number_unsat_literals = 0
            sat_clause = False
            for literal in clause:
                if abs(literal) in candidates:
                    if literal == candidates[abs(literal)] * abs(literal):
                        if not sat_clause:
                            sat_clause = True
                            number_sat_clauses += 1
                    else:
                        number_unsat_literals += 1
            if number_unsat_literals == len(clause):
                return 0
        if number_sat_clauses == len(formula):
            return 2
        return 1
