# Copyright (c) 2025 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# main author: Hannes Eberhard

import re
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from random import Random
from typing import Optional

from z3 import ArithRef, Distinct, ExprRef, Int, Or, Solver, sat

from .abstract_generator import AbstractGenerator
from .abstract_verifier import AbstractVerifier
from .einstein_attributes import Faker, FakerType


@dataclass
class SolverVariables:
    """
    Utility class to store the variables of a Z3 solver.
    Allows mapping attributes to their corresponding Z3 variables.

    The variables are stored in a 2-level dictionary, with the attribute type and value as keys.
    """

    variables: dict["AttributeType", dict[str, ArithRef]]

    def __init__(self) -> None:
        """
        Initialize the SolverVariables instance with an empty dictionary.
        """
        self.variables = defaultdict(dict)

    def add_attribute(self, attribute: "Attribute", val: ArithRef) -> None:
        """
        Add an attribute to the SolverVariables instance.

        Args:
            attribute (Attribute): Attribute to add.
            val (ArithRef): Z3 variable corresponding to the attribute.
        """
        self.variables[attribute.type][attribute.value] = val

    def get(self, attribute: "Attribute") -> ArithRef:
        """
        Get the Z3 variable corresponding to an attribute.

        Args:
            attribute (Attribute): Attribute to get the variable for.
        Returns:
            ArithRef: Z3 variable corresponding to the attribute.
        """
        return self.variables[attribute.type][attribute.value]


class AttributeType(str, Enum):
    """
    Enum class to define the different attribute types that can be used in the Einstein domain.
    """

    Position = "position"
    Name = "name"
    Nationality = "nationality"
    HouseColor = "house_color"
    Drink = "drink"
    Food = "food"
    Pet = "pet"
    Job = "job"
    Car = "car"

    def to_faker_type(self) -> Optional[FakerType]:
        """
        Map the attribute type to the corresponding FakerType, if available.

        Returns:
            Optional[FakerType]: Corresponding FakerType.
        """

        match self:
            case AttributeType.Name:
                return FakerType.Name
            case AttributeType.Nationality:
                return FakerType.Nationality
            case AttributeType.HouseColor:
                return FakerType.Color
            case AttributeType.Drink:
                return FakerType.Drink
            case AttributeType.Food:
                return FakerType.Food
            case AttributeType.Pet:
                return FakerType.Animal
            case AttributeType.Job:
                return FakerType.Job
            case AttributeType.Car:
                return FakerType.Car
            case _:
                return None

    def pretty(self) -> str:
        """
        Return a human-readable version of the attribute type.

        Returns:
            str: String for the attribute type.
        """

        match self:
            case AttributeType.Position:
                return "Position"
            case AttributeType.Name:
                return "Name"
            case AttributeType.Nationality:
                return "Nationality"
            case AttributeType.HouseColor:
                return "House Color"
            case AttributeType.Drink:
                return "Drink"
            case AttributeType.Food:
                return "Food"
            case AttributeType.Pet:
                return "Pet"
            case AttributeType.Job:
                return "Job"
            case AttributeType.Car:
                return "Car"


@dataclass(frozen=True)
class Attribute:
    """
    Data class to represent an attribute in the Einstein domain.
    Each attribute consists of a type and the corresponding value.

    `frozen=True` ensures that instances are hashable.
    """

    type: AttributeType
    value: str

    def pretty(self) -> str:
        """
        Return a human-readable version of the attribute.

        Returns:
            str: Pretty attribute representation.
        """
        return f"{self.value} ({self.type.pretty()})"


@dataclass
class EinsteinSolution:
    """
    Data class to represent a solution in the Einstein domain.
    A solution defines the position each attribute is assigned to.

    Additionally, the number of houses and attribute types are stored.
    """

    n: int
    attribute_types: list[AttributeType]
    solution: dict[AttributeType, list[str]]

    def __init__(
        self,
        n: int,
        attribute_types: list[AttributeType],
        solution: dict[AttributeType, list[str]],
    ) -> None:
        """
        Initialize the EinsteinSolution instance with the number of houses, attribute types, and the
        solution.

        Args:
            n (int): Number of houses.
            attribute_types (list[AttributeType]): List of attribute types.
            solution (dict[AttributeType, list[str]]): Solution dictionary mapping attribute types
                                                       to values.
        """
        self.n = n
        self.attribute_types = attribute_types
        self.solution = solution

    @staticmethod
    def generate(
        n: int = 5,
        attribute_types: list[AttributeType] = list(AttributeType),
        random: Random = Random(),
    ) -> "EinsteinSolution":
        """
        Generate a random solution for the Einstein domain.
        This method samples random values for each house and attribute type.

        Args:
            n (int): Number of houses. Defaults to 5.
            attribute_types (list[AttributeType]): List of attribute types to generate. Defaults to
                                                   all available types.
            random (Random): Random generator. Defaults to Random().
        Returns:
            EinsteinSolution: Random solution.
        Raises:
            ValueError: Raises an exception.
        """
        solution = dict()

        for attribute_type in attribute_types:
            faker_type = attribute_type.to_faker_type()

            if attribute_type == AttributeType.Position:
                # Houses are 1-indexed
                values = [str(i + 1) for i in range(n)]
                solution[attribute_type] = values
            elif faker_type:
                faker = Faker(faker_type, random)
                values = faker.sample(n)
                solution[attribute_type] = values
            else:
                raise ValueError(f"Unsupported attribute type: {attribute_type}")

        return EinsteinSolution(n, attribute_types, solution)

    def pretty(self) -> str:
        """
        Return a human-readable version of the solution.
        Format the solution as a table with the attribute types as columns and the values as rows.

        Returns:
            str: Pretty solution representation.
        """
        rows = []
        width = 15

        for attribute_type in self.attribute_types:
            values = self.solution[attribute_type]
            values = [value.ljust(width) for value in values]
            rows.append(f"{attribute_type.value.ljust(width)} | {' '.join(values)}")

        return "\n".join(rows)

    def pretty_attributes(self) -> str:
        """
        Return a human-readable version of the solution attributes.
        Format each attribute type with a list of its values.

        Returns:
            str: Pretty solution attributes representation.
        """
        rows = []

        for attribute_type, attributes in self.solution.items():
            if attribute_type == AttributeType.Position:
                continue

            rows.append(f"{attribute_type.pretty()}: {', '.join(sorted(attributes))}")

        return "\n".join(rows)


class ConstraintType(str, Enum):
    """
    Enum class to define the different constraint / clue types that can be used for an Einstein riddle.
    """

    Equal = "Equal"
    NotEqual = "NotEqual"
    LeftTo = "LeftTo"
    RightTo = "RightTo"
    NextTo = "NextTo"


@dataclass
class Constraint:
    """
    Data class to represent a constraint / clue in the Einstein domain.
    The relationship semantics are as follows:

    - (Equal, A, B): The house with attribute A is the same as the house with attribute B.
    - (NotEqual, A, B): The house with attribute A is different from the house with attribute B.
    - (LeftTo, A, B): The house with attribute A is directly to the left of the house with attribute B. The order is ... A, B, ...
    - (RightTo, A, B): The house with attribute A is directly to the right of the house with attribute B. The order is ... B, A, ...
    - (NextTo, A, B): The house with attribute A is adjacent to the house with attribute B. The order is ... A, B, ... or ... B, A, ...
    """

    type: ConstraintType
    source: Attribute
    target: Attribute

    @staticmethod
    def generate(solution: EinsteinSolution, random: Random) -> "Constraint":
        """
        Sample a random constraint for a given solution.
        The constraint type is uniformly randomly selected, and the source and target attributes are
        sampled accordingly.

        It is ensured that the source attribute is not Position which eliminates trivial constraints
        such as "The house with attribute Position 1 is directly to the left of the house with
        attribute Position 2".

        Args:
            solution (EinsteinSolution): Solution to generate the constraint for.
            random (Random): Random generator.
        Returns:
            Constraint: Random constraint that holds for the solution.
        """

        # Generate the indices to sample the source and target attributes from
        idx = list(range(solution.n))
        left_idx = idx[:-1]
        right_idx = idx[1:]

        constraint_type = random.choice(list(ConstraintType))

        match constraint_type:
            case ConstraintType.Equal:
                source_idx = random.choice(idx)
                target_idx = source_idx
            case ConstraintType.NotEqual:
                source_idx = random.choice(idx)
                target_idx = random.choice(idx)
                while source_idx == target_idx:
                    target_idx = random.choice(idx)
            case ConstraintType.LeftTo:
                source_idx = random.choice(left_idx)
                target_idx = source_idx + 1
            case ConstraintType.RightTo:
                source_idx = random.choice(right_idx)
                target_idx = source_idx - 1
            case ConstraintType.NextTo:
                match random.choice([-1, 1]):
                    case -1:
                        source_idx = random.choice(left_idx)
                        target_idx = source_idx + 1
                    case 1:
                        source_idx = random.choice(right_idx)
                        target_idx = source_idx - 1

        source_attribute = random.choice(solution.attribute_types)

        # Ensure source attribute is not Position
        while source_attribute == AttributeType.Position:
            source_attribute = random.choice(solution.attribute_types)

        target_attribute = random.choice(solution.attribute_types)

        constraint = Constraint(
            constraint_type,
            Attribute(
                source_attribute, solution.solution[source_attribute][source_idx]
            ),
            Attribute(
                target_attribute, solution.solution[target_attribute][target_idx]
            ),
        )

        return constraint

    def pretty(self) -> str:
        """
        Return a human-readable version of the constraint.

        Returns:
            str: Pretty constraint representation.
        """

        match self.type:
            case ConstraintType.Equal:
                return f"{self.source.pretty()} is equal to {self.target.pretty()}"
            case ConstraintType.NotEqual:
                return f"{self.source.pretty()} is not equal to {self.target.pretty()}"
            case ConstraintType.LeftTo:
                return f"{self.source.pretty()} is directly to the left of {self.target.pretty()}"
            case ConstraintType.RightTo:
                return f"{self.source.pretty()} is directly to the right of {self.target.pretty()}"
            case ConstraintType.NextTo:
                return f"{self.source.pretty()} is adjacent to {self.target.pretty()}"

    def to_z3(self, variables: SolverVariables) -> ExprRef:
        """
        Convert the constraint to a Z3 expression.
        Each variable represents the integer index of the house with the corresponding attribute.

        Z3 expressions are instantiated based on the semantics mentioned above.

        Args:
            variables (SolverVariables): Variables of the Z3 solver.
        Returns:
            ExprRef: Z3 expression representing the constraint.
        """

        source = variables.get(self.source)
        target = variables.get(self.target)

        match self.type:
            case ConstraintType.Equal:
                return source == target
            case ConstraintType.NotEqual:
                return source != target
            case ConstraintType.LeftTo:
                return source == target - 1
            case ConstraintType.RightTo:
                return source == target + 1
            case ConstraintType.NextTo:
                return Or(source == target - 1, source == target + 1)


@dataclass
class EinsteinRiddle:
    """
    Data class to represent an Einstein riddle.
    The riddle mainly consists of a list of constraints that allow solving the riddle uniquely.
    """

    n: int
    constraints: list[Constraint]
    variables: SolverVariables
    solver: Solver
    target: Optional[tuple[AttributeType, int]]

    def __init__(
        self,
        n: int,
        constraints: list[Constraint],
        variables: SolverVariables,
        target: Optional[tuple[AttributeType, int]],
    ) -> None:
        """
        Initialize the `EinsteinRiddle` instance with the number of houses, constraints, variables, and a target attribute.

        Args:
            n (int): Number of houses.
            constraints (list[Constraint]): List of constraints.
            variables (SolverVariables): Variables of the Z3 solver.
            target (Optional[tuple[AttributeType, int]]): Target attribute and position.
        """

        self.n = n
        self.constraints = constraints
        self.variables = variables
        self.solver = self.init_solver()
        self.target = target

    @staticmethod
    def generate(
        solution: EinsteinSolution,
        random: Random = Random(),
    ) -> "EinsteinRiddle":
        """
        Generate a random riddle with the given solution.

        In the first phase, constraints are randomly sampled until a unique riddle is found.
        In the second phase, constraints are removed one by one until the riddle remains unique.
        This ensures that the riddle is uniquely solvable and non-trivial.

        Lastly, a target attribute is selected that will be used to formulate the main question of
        the riddle, e.g. "Which house is associated with John?". Preference is give to attributes
        that are not used in the constraints to ensure the questions require solving the entire
        riddle. If this is not possible, a random attribute is selected (excluding position
        attributes).

        Args:
            solution (EinsteinSolution): Solution to generate the riddle for.
            random (Random): Random generator. Defaults to Random().
        Returns:
            EinsteinRiddle: Random Einstein riddle.
        """

        constraints: list[Constraint] = list()
        variables = SolverVariables()

        # Initialize the variables with the solution attributes
        for attribute_type, values in solution.solution.items():
            for value in values:
                attribute = Attribute(attribute_type, value)
                variables.add_attribute(
                    attribute, Int(f"{attribute_type.value}_{value}")
                )

        while True:
            # Add random constraints in steps of 20
            for _ in range(20):
                constraint = Constraint.generate(solution, random)
                constraints.append(constraint)

            # Check if the riddle is uniquely solvable
            riddle = EinsteinRiddle(solution.n, constraints, variables, None)
            if riddle.is_unique(solution):
                break

        # Constraint i is included iff mask[i] == True
        mask = [True] * len(constraints)

        for i, constraint in enumerate(constraints):
            # Check if the riddle remains unique when removing constraint i
            mask[i] = False
            filtered = [c for c, m in zip(constraints, mask) if m]
            riddle = EinsteinRiddle(solution.n, filtered, variables, None)

            if not riddle.is_unique(solution):
                mask[i] = True

        constraints = [c for c, m in zip(constraints, mask) if m]

        all_attributes: set[Attribute] = set()

        for attribute_type, values in solution.solution.items():
            if attribute_type == AttributeType.Position:
                continue

            for value in values:
                all_attributes.add(Attribute(attribute_type, value))

        used_attributes: set[Attribute] = set()

        for constraint in constraints:
            for attribute in [constraint.source, constraint.target]:
                used_attributes.add(attribute)

        unused_attributes = all_attributes - used_attributes

        # If all attributes are used, add a new constraint which doesn't involve position attributes
        if not unused_attributes:
            for constraint in constraints:
                if (
                    constraint.source.type == AttributeType.Position
                    or constraint.target.type == AttributeType.Position
                ):
                    continue

                unused_attributes.add(constraint.source)
                unused_attributes.add(constraint.target)

        unused_attributes = list(unused_attributes)
        unused_attributes.sort(key=lambda x: (x.type, x.value))

        target_attribute = random.choice(unused_attributes)
        target_position = (
            solution.solution[target_attribute.type].index(target_attribute.value) + 1
        )

        return EinsteinRiddle(
            solution.n, constraints, variables, (target_attribute, target_position)
        )

    def init_solver(self) -> Solver:
        """
        Initialize the Z3 solver with the variables and constraints of the riddle.
        Besides the attribute constraints, the solver is also initialized with the semantics of the
        houses (distinct and within the range [0, n)).

        Returns:
            Solver: Initialized Z3 solver.
        """

        solver = Solver()

        for values in self.variables.variables.values():
            solver.add(Distinct(*values.values()))

            for variable in values.values():
                solver.add(variable >= 0)
                solver.add(variable < self.n)

        for constraint in self.constraints:
            solver.add(constraint.to_z3(self.variables))

        return solver

    def solve(self) -> Optional[EinsteinSolution]:
        """
        Solve the riddle and return the solution, if one was found.

        Returns:
            Optional[EinsteinSolution]: Solution to the riddle, if one was found.
        """
        if self.solver.check() != sat:
            return None

        solution = {
            attribute_type: [""] * self.n for attribute_type in self.variables.variables
        }
        model = self.solver.model()

        for attribute_type, values in self.variables.variables.items():
            for value, variable in values.items():
                idx = model[variable].as_long()
                solution[attribute_type][idx] = value

        return EinsteinSolution(self.n, list(solution.keys()), solution)

    def is_unique(self, solution: EinsteinSolution) -> bool:
        """
        Check if a riddle has an unique solution and that solution matches the provided solution.

        Args:
            solution (EinsteinSolution): Solution to check against.
        Returns:
            bool: True iff the riddle has a unique solution that matches the provided solution.
        """
        solver = self.init_solver()
        disjunction = []

        for attribute_type, values in solution.solution.items():
            idx_map = {value: i for i, value in enumerate(values)}

            for value in values:
                attribute = Attribute(attribute_type, value)
                variable = self.variables.get(attribute)
                idx = idx_map[value]
                disjunction.append(variable != idx)

        return solver.check(Or(disjunction)) != sat

    def pretty_constraints(self) -> str:
        """
        Return a human-readable version of the constraints.
        Format the constraints as a numbered list.

        Returns:
            str: Pretty constraints representation.
        """

        rows = []

        for idx, constraint in enumerate(self.constraints):
            rows.append(f"{idx + 1}. {constraint.pretty()}")

        return "\n".join(rows)

    def pretty_target(self) -> Optional[str]:
        """
        Return a human-readable version of the target attribute, if it is set.
        Format the target attribute as a question, e.g. "Which house is associated with John?".

        Returns:
            Optional[str]: Pretty target attribute representation if the target is set. Otherwise, None.
        """
        if not self.target:
            return None

        target_attribute, _ = self.target
        return f"Which house is associated with {target_attribute.pretty()}?"


class EinsteinGenerator(AbstractGenerator):
    """
    The EinsteinGenerator class generates tasks within the Einstein domain.

    Inherits from the AbstractGenerator class and implements its abstract methods.
    """

    def __init__(self, seed: int | None = None) -> None:
        """
        Initialize the EinsteinGenerator instance with a seed for the random number generation as well
        as other necessary data structures.

        Args:
            seed (int | None): Seed for the random number generation. Defaults to None.
        """
        super().__init__(seed)

    def map_difficulty(
        self,
        difficulty: float,
        min_houses: int = 3,
        max_houses: int = 7,
        min_attributes: int = 3,
        max_attributes: int = 7,
    ) -> tuple[int, int]:
        """
        Map the difficulty level to the number of houses and attributes for the Einstein task.

        Number of houses scales linearly with difficulty.
        Number of attributes scales linearly with difficulty for each value of n_houses.

        Args:
            difficulty (float): Difficulty level in [0, 1].
            min_houses (int): Minimum number of houses. Defaults to 3.
            max_houses (int): Maximum number of houses. Defaults to 7.
            min_attributes (int): Minimum number of attributes. Defaults to 3.
            max_attributes (int): Maximum number of attributes. Defaults to 7.

        Returns:
            tuple[int, int]: Number of houses and attributes.
        """

        # Inclusive range
        max_houses += 1
        max_attributes += 1

        # Limit attributes to the number of available attribute types
        max_attributes = min(max_attributes, len(AttributeType))

        n_house_steps = max_houses - min_houses
        raw_n_houses = int(difficulty * n_house_steps)
        n_houses = min_houses + raw_n_houses

        attributes_difficulty = (difficulty * n_house_steps) % 1
        n_attribute_steps = max_attributes - min_attributes
        raw_n_attributes = int(attributes_difficulty * n_attribute_steps)
        n_attributes = min_attributes + raw_n_attributes

        return n_houses, n_attributes

    def generate(self, difficulty: float) -> dict:
        """
        Generate a task in the Einstein riddle domain.

        The difficulty determines the number of houses and attributes for the Einstein riddle.

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

        n_houses, n_attributes = self.map_difficulty(difficulty)

        # Select "Position" attribute type and n_attributes - 1 random attribute types
        attribute_types = [AttributeType.Position] + self.random.sample(
            list(AttributeType)[1:], n_attributes - 1
        )

        solution = EinsteinSolution.generate(
            n_houses, attribute_types=attribute_types, random=self.random
        )

        riddle = EinsteinRiddle.generate(solution, random=self.random)
        _, target_position = riddle.target

        pretty_attributes = solution.pretty_attributes()
        pretty_constraints = riddle.pretty_constraints()
        pretty_target = riddle.pretty_target()

        self.solution = target_position

        problem = (
            f"There are {n_houses} houses in a row (numbered 1 to {n_houses}).\n"
            f"\n"
            f"Each house is occupied by a a different person and each house/person has an unique attribute from the following sets:\n"
            f"{pretty_attributes}\n"
            f"\n"
            f"You are given the following information:\n"
            f"{pretty_constraints}\n"
            f"\n"
            f"{pretty_target} Format your answer as \\boxed{{x}} where x is the number of the house."
        )

        task = {
            "type": "Einstein",
            "level": int(difficulty * 10) + 1,
            "problem": problem,
            "unique_solution": True,
            "solution": str(target_position),
        }

        return task


class EinsteinVerifier(AbstractVerifier):
    """
    The EinsteinVerifier class verifies tasks and potential solutions within the Einstein riddle
    domain.

    Inherits from the AbstractVerifier class and implements its abstract methods.
    """

    def verify(self, task: dict, output: str) -> int:
        """
        Verify the output of a model for a given task.

        The output is scored with the following scheme:
        -1: The output is invalid.
         0: The output is incorrect.
         2: The problem was solved.

        Args:
            task (dict): Task description as returned by the generator.
            output (str): Model output.
        Returns:
            int: Score for the output.
        """

        answer = self.parse(output)

        if answer is None:
            return -1

        return 2 if answer == task["solution"] else 0

    def parse(self, output: str) -> Optional[str]:
        """
        Parse the output of a model and extract the answer.
        The models are instructed to format the answer as "\\boxed{x}" where x is the number of the
        house. If the answer cannot be extracted, None is returned, otherwise the extracted answer
        is returned.

        Args:
            output (str): Model output.
        Returns:
            Optional[str]: Extracted answer if possible, otherwise None.
        """

        if "boxed" not in output:
            return None

        match = re.search(r"boxed{(\d+)}", output)

        if not match:
            return None

        return match.group(1)
