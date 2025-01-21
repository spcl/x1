# Copyright (c) 2025 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# main authors:
# Eric Schreiber
# Julia Barth
#
# contributions: Afonso Catarino

import concurrent.futures
import json
import logging
import os
import time
import uuid
from multiprocessing import get_context
from typing import Any, Iterator, Optional

import torch
from torchtyping import TensorType
from tqdm import tqdm

from x1.evaluation import AbstractStepEvaluation
from x1.expansion import AbstractStepExpansion

from .utils import MinMaxStats

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

"""
args is used for both the Node class as well as the MCTS class.

format of args:
{
    "alpha": float # Weight of the simulation value. Defaults to 0.5 if simulations are used, 0 otherwise.
    "c1": float # Constant to control exploration. Defaults to 1.25.
    "c2": float # Constant to control exploration. Defaults to 19652.
    "debug": bool # Flag to indicate, whether to output debug information. Defaults to False.
    "gamma": float # Discounting variable for trajectories. Defaults to 0.96.
    "gt_solution": str # Ground truth solution. Defaults to None.
    "lr_td": float # Learning rate for TD-MCTS. Defaults to 1.
    "max_search_iterations": int # Maximum number of expansion steps during the MCTS.
    "num_new_children": int # Number of child nodes that will be generated during the expansion. Defaults to 2.
    "num_simulations_per_node": int # Number of simulations per node. Defaults to 0.
    "output_path": str # Path of the directory, where the MCTS tree will be stored. Defaults to "graphs".
    "policy_api_url": str # URL to access the policy model.
    "question": str # Task to solve.
    "reward_function": Callable[[[Optional[Any], str, list[float]], float] # Function to compute rewards.
    "reward_range": list[float] # Minimum and maximum of the range for the reward. Defaults to [-1, 1].
    "save_tree": bool # Flag to indicate, whether to write MCTS tree to a file. Defaults to False.
    "stop_if_terminal_found": bool # Stop the MCTS as soon as a terminal node is found. Defaults to False.
    "value_api_url": str # URL to access the value model.
}
"""


class Node:
    """
    Class representing a node in the MCTS tree.
    """

    def __init__(
        self,
        args: dict,
        parent: Optional["Node"] = None,
        reasoning_step: Optional[Any] = None,
        visit_count: int = 0,
        reward: float = 0.0,
        q_value_model: float = 0.0,
        q_value: float = 0.0,
        simulation_value: float = 0.0,
        simulations: list[Any] = [],
        advantage: float = 0.0,
        is_terminal: bool = False,
        contextual_memory: Optional[list[Any]] = None,
        children: list["Node"] = [],
        leads_to_terminal: bool = False,
    ) -> None:
        """
        Initialize a node in the MCTS tree.

        Args:
            args (dict): Configuration for the MCTS search.
            parent (Optional[Node]): Parent node in the tree. None for the root node. Defaults to None.
            reasoning_step (Any): Reasoning step representing this node. Defaults to None.
            visit_count (int): Number of times this node has been visited. Defaults to 0.
            reward (float): Reward associated with reasoning_step. Defaults to 0.0.
            q_value_model (float): Initial prediction of the quality value. Defaults to 0.0.
            q_value (float): Quality value. Defaults to 0.0.
            simulation_value (float): Average value of the simulations. Defaults to 0.0.
            simulations (list[Any]): List of simulations starting from reasoning_step. Defaults to [].
            advantage (float): Advantage of the node. Defaults to 0.0.
            is_terminal (bool): True if the node is a terminal node. Defaults to False.
            contextual_memory (list[Any]): Context up to reasoning_step. Defaults to None.
            children (list[Node]): List of children nodes. Defaults to [].
            leads_to_terminal (bool): Flag to indicate whether the subtree below the node contains
                                      any terminal nodes. Defaults to False.
        """

        self.args = args

        # MCTS configuration
        # Set the ground truth solution
        if "gt_solution" in args:
            self.gt_solution = args["gt_solution"]
        else:
            self.gt_solution = None
        # Set num_simulations
        if "num_simulations_per_node" in args:
            assert (
                self.gt_solution is not None
            ), "num_simulations_per_node is set, but no gt_solution is provided. You cannot get a value for the simulation."
            self.num_simulations = args["num_simulations_per_node"]
        else:
            self.num_simulations = 0

        # Set the range of the reward
        if "reward_range" in args:
            self.reward_range = args["reward_range"]
        else:
            self.reward_range = [-1, 1]

        # Set the reward function
        if "reward_function" in args:
            self.reward_function = args["reward_function"]
        elif "reward_range" not in args and self.num_simulations > 0:
            raise ValueError("Reward function not set in args")

        # Set alpha, which describes the weight of the simulation value
        if "alpha" in args:
            self.alpha = args["alpha"]
        elif "alpha" not in args and self.num_simulations > 0:
            self.alpha = 0.5  # Weighing factor for reasoning step and simulation values
        else:
            self.alpha = 0  # No simulation

        # Graph structure
        self.id = str(uuid.uuid4())
        self.parent = parent
        self.reasoning_step = reasoning_step  # New reasoning step
        self.contextual_memory = (
            contextual_memory if contextual_memory is not None else []
        )  # The contextual memory is storing previous reasoning steps.

        self.is_terminal = is_terminal

        # Graph property
        self.level = 0 if self.parent is None else self.parent.level + 1

        # Node statistics: [visit count, value sum, policy probability] + q-value
        self.visit_count = visit_count  # N(s)
        self.reward = reward
        self.q_value = q_value  # Q(s,a) for TD-MCTS approximation
        self.q_value_model = (
            q_value_model  # Save what the model has predicted initially
        )
        self.gamma = args.get("gamma", 0.96)  # Discounting variable for trajectories
        self.lr_td = args.get("lr_td", 1)  # Learning rate for TD-MCTS
        self.advantage = advantage

        # Simulation
        self.simulation_value = simulation_value
        self.simulations = simulations

        # Children nodes
        self.children = children

        self.leads_to_terminal = leads_to_terminal

    def is_expanded(self) -> bool:
        """
        Check if the node has already been expanded.

        Returns:
            bool: True if node has children
        """
        return len(self.children) > 0

    def select(
        self, min_max_stats: MinMaxStats, c1: float = 1.25, c2: float = 19652
    ) -> "Node":
        """
        Return child node with the highest selection score.

        Args:
            min_max_stats (MinMaxStats): Object to keep track of min and max values for normalization.
            c1 (float): Constant to control exploration. Defaults to 1.25.
            c2 (float): Constant to control exploration. Defaults to 19652.
        Returns:
            Node: Best child node.
        """
        selection_score_values = self.get_selection_score(min_max_stats, c1=c1, c2=c2)
        best_child_idx = torch.argmax(selection_score_values).item()
        return self.children[best_child_idx]

    def get_selection_score(
        self, min_max_stats: MinMaxStats, c1: float = 1.25, c2: float = 19652
    ) -> TensorType["num_new_children"]:
        """
        Return the selection scores for all child nodes.
        Currently implements a nuanced UCB scored similar to AlphaZero's PUCT formula.

        Args:
            min_max_stats (MinMaxStats): Object to keep track of min and max values for normalization.
            c1 (float): Constant to control exploration. Defaults to 1.25.
            c2 (float): Constant to control exploration. Defaults to 19652.
        Returns:
            TensorType["num_of_samples"]: Selection score for each child.
        """
        # List to tensor
        own_visit_count = torch.tensor(self.visit_count, dtype=torch.float)
        children_visit_count = torch.tensor(
            [child.visit_count for child in self.children], dtype=torch.float
        )
        children_q_value = torch.tensor(
            [child.q_value for child in self.children], dtype=torch.float
        )

        # Compute selection score

        # UCB based scoring
        exploration_term = (
            torch.sqrt(own_visit_count - 1) / (1 + children_visit_count)
        ) * (c1 + torch.log((own_visit_count - 1 + c2 + 1) / c2))
        # Value term, which means here the mean value of the child
        # min_max_stats.normalize: Normalizes the average value to [0,1] range, making sure the c1
        # and c2 values are the same for different reward settings.
        value_term = min_max_stats.normalize(children_q_value)

        selection_score = value_term + exploration_term
        return selection_score

    def simulate(self, expansion: AbstractStepExpansion, num_simulations: int) -> float:
        """
        Generate simulations for the node, starting from 'reasoning_step'.

        Args:
            expansion (AbstractStepExpansion): Class to generate simulations.
            num_simulations (int): Number of simulations to generate.
        Returns:
            float: Average reward of the simulations.
        """
        self.simulations = expansion.simulate_end(
            contextual_memory=self.contextual_memory + [self.reasoning_step],
            num_simulations=num_simulations,
        )
        self.simulation_value = (
            sum(
                [
                    self.reward_function(
                        output_str=simulation,
                        gt_string=self.gt_solution,
                        reward_range=self.reward_range,
                    )
                    for simulation in self.simulations
                ]
            )
            / num_simulations
        )
        return self.simulation_value

    def simulate_and_evaluate(
        self,
        expansion: AbstractStepExpansion,
        evaluation: AbstractStepEvaluation,
        is_terminal: bool,
    ) -> float:
        """
        Generate simulations and evaluate 'reasoning_step' using the evaluation transformation.

        Args:
            expansion (AbstractStepExpansion): Class to generate simulations.
            evaluation (AbstractStepEvaluation): Class to evaluate the reasoning steps.
            is_terminal (bool): True if the node is a terminal node.
        Returns:
            float: Average reward of the simulations.
        """
        if self.alpha < 1:
            self.q_value_model = evaluation.evaluate_step(
                self.contextual_memory, self.reasoning_step, is_terminal
            )

        if self.num_simulations == 0 or self.alpha == 0 or is_terminal:  # No simulation
            self.q_value = self.q_value_model
            return self.q_value

        else:  # Simulation
            # Make a CoT prediction towards the end of the game
            self.simulation_value = self.simulate(
                expansion=expansion, num_simulations=self.num_simulations
            )
            self.q_value = (
                1 - self.alpha
            ) * self.q_value_model + self.alpha * self.simulation_value
            return self.q_value

    def expand_and_evaluate(
        self,
        expansion: AbstractStepExpansion,
        evaluation: AbstractStepEvaluation,
        num_samples: int = 2,
        max_rewards: int = 100,
        max_processes: int = 16,
    ) -> None:
        """
        Expand the node by generating children with the next reasoning steps and evaluating them.
        Parallelizes the generation of children up to 'max_processes' processes.

        Args:
            expansion (AbstractStepExpansion): Class to generate reasoning steps.
            evaluation (AbstractStepEvaluation): Class to evaluate reasoning steps.
            num_samples (int): Number of samples to generate. Defaults to 2.
            max_rewards (int): Maximum reward value. Defaults to 100.
            max_processes (int): Maximum number of processes to use. Defaults to 16.
        """

        # generate_steps returns a list of tuples (step, is_final)
        steps_tuple_list = expansion.generate_steps(
            self.contextual_memory + [self.reasoning_step], num_steps=num_samples
        )

        # If no valid actions remain
        if len(steps_tuple_list) == 0:
            return

        # Parallelize the following part with multiprocessing: use num_samples processes but at most max_processes
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=min(num_samples, max_processes), mp_context=get_context("spawn")
        ) as executor:
            futures = [
                executor.submit(
                    create_children_and_eval,
                    self,
                    expansion,
                    evaluation,
                    step,
                    is_final,
                )
                for step, is_final in steps_tuple_list
            ]
            children: list[Node] = []
            for future in concurrent.futures.as_completed(futures):
                try:
                    children.append(future.result(timeout=180))  # Timeout in seconds
                except concurrent.futures.TimeoutError:
                    print("A future timed out.", flush=True)
                except Exception as e:
                    print(f"A future raised an exception: {e}", flush=True)

        # Sequential execution starts here again
        for child in children:
            child.parent = self

            # Update children memory
            self.children = self.children + [child]

    def backpropagate(self) -> None:
        """
        Update the q value of the node and recursively backpropagate that q value to the parent
        node.

        Theoretically, the q value of the node should be updated as follows:
        Q(s) = (1 - lr_td) * Q(s) + lr_td * (SUM_i (N(s_chidld_i) + 1) * (gamma * Q(s_child_i)
               + reward_child_i)) / SUM_j (N(s_child_j) + 1)
        Because we have only final rewards, we will never do a backpropagation with a child that
        has a non-zero reward. So we can simplify the formula to:
        Q(s) = (1 - lr_td) * Q(s) + lr_td * (SUM_i (N(s_chidld_i) + 1) * gamma * Q(s_child_i))
               / SUM_j (N(s_child_j) + 1)
        """
        value_sum = 0.0
        children_total_visit_count_sum = 0
        for child in self.children:
            value_sum += child.visit_count * self.gamma * child.q_value
            children_total_visit_count_sum += child.visit_count

        # Because we have only final rewards, we do not need to consider the reward below
        self.q_value = (
            1 - self.lr_td
        ) * self.q_value + self.lr_td * value_sum / children_total_visit_count_sum

        self.visit_count += 1

        # Set leads_to_terminal flag if any child leads to terminal node
        self.leads_to_terminal = any(
            [child.leads_to_terminal for child in self.children]
        )

        if self.parent is not None:  # Not root
            self.parent.backpropagate()

    def rec_dict(self) -> dict:
        """
        Return a dictionary representation of the node.

        Returns:
            dict: Dictionary representation of the node.
        """
        node_dict = {
            "id": self.id,
            "reasoning_step": self.reasoning_step,
            "contextual_memory": self.contextual_memory,
            "parent": (self.parent.id if self.parent else None),
            "visit_count": self.visit_count,
            "reward": self.reward,
            "q_value_model": self.q_value_model,
            "q_value": self.q_value,
            "is_terminal": self.is_terminal,
            "leads_to_terminal": self.leads_to_terminal,
            "num_simulations": self.num_simulations,
            "alpha": self.alpha,
            "simulation_value": (
                self.simulation_value if self.num_simulations > 0 else None
            ),
            "simulations": self.simulations if self.num_simulations > 0 else None,
            "children": [child.rec_dict() for child in self.children],
        }
        return node_dict

    def save_to_json(self, filepath: str) -> None:
        """
        Store the entire tree structure (from the root) in a JSON file.

        Args:
            filepath (str): Path to the file, where the tree structure will be stored.
        """
        with open(filepath, "w") as f:
            _dict = self.rec_dict()
            _dict["args"] = self.args
            json.dump(_dict, f, indent=4, default=lambda _: "<not serializable>")


class MCTS:
    def __init__(
        self,
        expansion: AbstractStepExpansion,
        evaluation: AbstractStepEvaluation,
        args: dict,
    ) -> None:
        """
        Initialize the MCTS search.

        Args:
            expansion (AbstractStepExpansion): Class to generate reasoning steps.
            evaluation (AbstractStepEvaluation): Class to evaluate reasoning steps.
            args (dict): Configuration for the MCTS search.
        """
        self.expansion = expansion
        self.evaluation = evaluation
        self.args = args
        self.set_class_variables_from_args(args)

    def select_best_path_selection(
        self, node: Node, min_max_stats: MinMaxStats, c1=1.25, c2=19652
    ) -> list[str]:
        """
        Select the best path according to the selection criteria.

        Args:
            node (Node): Node to start the selection from.
            min_max_stats (MinMaxStats): Object to keep track of min and max values for normalization.
            c1 (float): Constant to control exploration. Defaults to 1.25.
            c2 (float): Constant to control exploration. Defaults to 19652.
        Returns:
            list[str]: List of reasoning steps that constitute the best path.
        """
        output = [node.reasoning_step]
        while node.is_expanded():
            node = node.select(min_max_stats, c1=c1, c2=c2)
            output.append(node.reasoning_step)

        return output

    def select_best_path_best_q_value(self, node: Node) -> list[str]:
        """
        Select the best path according to the q value.

        Args:
            node (Node): Node to start the selection from.
        Returns:
            list[str]: List of reasoning steps that constitute the best path.
        """
        output = [node.reasoning_step]
        while node.is_expanded():
            node = max(node.children, key=lambda x: x.q_value)
            output.append(node.reasoning_step)

    def traverse_tree(self, node: Node) -> Iterator[Node]:
        """
        Traverse the tree below node in a DFS manner.

        Args:
            node (Node): Node to start the traversal from.
        Returns:
            Iterator[Node]: Iterator over the nodes in the tree.
        """
        # Use a stack for DFS
        stack = [node]
        while stack:
            current_node = stack.pop()
            yield current_node
            if current_node.is_expanded():  # No cycles allowed
                stack.extend(current_node.children)

    def check_args(self, args: dict) -> bool:
        """
        Check whether args contains all necessary arguments for the configuration.

        Args:
            args (dict): Arguments to check.
        Returns:
            bool: Whether all necessary arguments are present.
        """
        supported_args = [
            "question",
            "num_new_children",
            "gt_solution",
            "reward_function",
            "num_simulations_per_node",
            "reward_range",
            "alpha",
            "save_tree",
            "output_path",
            "max_search_iterations",
            "c1",
            "c2",
            "stop_if_terminal_found",
            "policy_api_url",
            "value_api_url",
            "debug",
            "gamma",
            "lr_td",
        ]
        # Check that all the args are supported
        for key in args:
            if key not in supported_args:
                print(
                    f"Argument {key} is not supported. Supported arguments are {supported_args}.",
                    flush=True,
                )

        # Check that there is a question to answer
        assert (
            "question" in args
        ), "No question provided. You need to provide a question to answer."

        # Check that a maximum number of search iterations is provided
        assert (
            "max_search_iterations" in args
        ), "No maximum number of search iterations provided. You need to provide a maximum number of search iterations."

        # If save_tree is set, check that the output_path is set
        if "save_tree" in args and args["save_tree"]:
            assert (
                "output_path" in args
            ), "save_tree is set but no output_path is provided. You need to provide a path to save the tree."

        # Check that if you do simulations, you have a ground truth solution and a reward function
        if "num_simulations_per_node" in args and args["num_simulations_per_node"] > 0:
            assert (
                "gt_solution" in args
            ), "num_simulations_per_node is set but no gt_solution is provided. You cannot get a value for the simulation."
            assert (
                "reward_function" in args
            ), "num_simulations_per_node is set but no reward_function is provided. You cannot get a value for the simulation."

        return True

    def set_class_variables_from_args(self, args: dict) -> None:
        """
        Set the number of new children to generate.

        Args:
            args (dict): Configuration for the MCTS search.
        """
        if "num_new_children" in args:
            self.num_new_children = args["num_new_children"]
        else:
            self.num_new_children = 2

    @torch.no_grad()
    def run_mcts(self, search_args: dict) -> tuple[list[str], Node]:
        """
        Start the MCTS from the root node.

        Args:
            search_args (dict): Configuration for the search.
        Returns:
            tuple[list[str], Node]: Tuple of the best path and the root node.
        """
        # Merge the MCTS args with the search args. If there are overlapping keys, the search args will overwrite the MCTS args
        self.args = {**self.args, **search_args}
        # Check that args contains a valid configuration
        assert self.check_args(self.args), "The args provided don't make sense."
        self.set_class_variables_from_args(self.args)

        if self.args.get("debug", False):
            print(f"Starting MCTS search with args: {self.args}", flush=True)

        if "reward_range" in self.args:
            min_max_stats = MinMaxStats(
                maximum=self.args["reward_range"][1],
                minimum=self.args["reward_range"][0],
            )
        else:
            min_max_stats = MinMaxStats(maximum=1, minimum=-1)

        if "output_path" in self.args:
            output_path = self.args["output_path"]
        else:
            output_path = "graphs"
        os.makedirs(output_path, exist_ok=True)

        # Check if empty
        if not os.listdir(output_path):
            pass
        else:
            # Create a subfolder with the current timestamp
            output_path = os.path.join(output_path, str(time.time()))
            os.makedirs(output_path, exist_ok=True)

        c1 = self.args.get("c1", 1.25)
        c2 = self.args.get("c2", 19652)
        stop_if_terminal_found = self.args.get("stop_if_terminal_found", False)
        save_tree = self.args.get("save_tree", False)

        self.root = Node(
            args=self.args, reasoning_step=self.args["question"], is_terminal=False
        )

        node = self.root
        logging.info(
            "Starting MCTS search with root reasoning step: '%s'", node.reasoning_step
        )

        max_iterations = self.args["max_search_iterations"]
        with tqdm(total=max_iterations, desc="MCTS Iterations", unit="iter") as pbar:
            for iteration in range(max_iterations):
                node = self.root

                # Select
                while node.is_expanded():
                    node = node.select(min_max_stats, c1=c1, c2=c2)

                # Expand
                if node.is_terminal:
                    logging.info("Terminal node reached: '%s'", node.reasoning_step)
                    break

                node.expand_and_evaluate(
                    self.expansion,
                    self.evaluation,
                    max_rewards=min_max_stats.maximum,
                    num_samples=self.num_new_children,
                )

                # Break if a single terminal node is found not if the terminal node is found during the next iteration
                if stop_if_terminal_found and any(
                    [child.is_terminal for child in node.children]
                ):
                    logging.info(
                        "Terminal node found directly during expansion: '%s'",
                        node.reasoning_step,
                    )
                    break

                # Backpropagate
                node.backpropagate()

                pbar.update(1)

            # Select the best path according to the reward_pi
            best_path = self.select_best_path_selection(
                self.root, min_max_stats, c1=c1, c2=c2
            )
            logging.info("Best path selected: %s", " -> ".join(best_path))

            if output_path is not None and save_tree:
                self.root.save_to_json(
                    os.path.join(output_path, "mcts_tree_final.json")
                )

            return best_path, self.root


# create_children_and_eval is not part of the Node class, so that it can be executed concurrently
def create_children_and_eval(
    parent_node: Node,
    expansion: AbstractStepExpansion,
    evaluation: AbstractStepEvaluation,
    next_step: str,
    is_terminal: bool,
) -> Node:
    """
    Create a child node from the parent node and evaluate it.

    Args:
        parent_node (Node): Parent node in the tree.
        expansion (AbstractStepExpansion): Class to generate reasoning steps.
        evaluation (AbstractStepEvaluation): Class to evaluate reasoning steps.
        next_step (str): Reasoning step representing the child node.
        is_terminal (bool): True if the child node is terminal.
    Returns:
        Node: Child node.
    """
    try:
        child = Node(
            parent_node.args,
            parent=parent_node,
            reasoning_step=next_step,
            contextual_memory=parent_node.contextual_memory
            + [parent_node.reasoning_step],
            visit_count=1,
            is_terminal=is_terminal,
        )

        if is_terminal and child.gt_solution is not None:
            child.reward = child.reward_function(
                output_str=next_step,
                gt_string=child.gt_solution,
                reward_range=child.reward_range,
            )
            child.q_value = child.reward  # + child.gamma * child.v_value (is 0 anyway)
        else:
            child.q_value = child.simulate_and_evaluate(
                expansion, evaluation, is_terminal=is_terminal
            )

        if is_terminal:
            child.leads_to_terminal = True

        return child
    except Exception as e:
        logging.error(f"Error in create_children_and_eval: {e}")
        raise e
