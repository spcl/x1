# Reasoning Structure and Strategy

This component implements a reasoning strategy based on Monte Carlo Tree Search (MCTS), which influences the chosen reasoning structure, that is also implemented by this component.


## Reasoning Structure: Node Class

The reasoning structure is implemented as a class called Node, which implements a node in the tree used for the MCTS and can be found in the file [mcts.py](mcts.py#L63).

### Attributes
In addition to the [args](#configuration) argument that provides the general configuration for the whole class, each nodes stores a number of attributes:
* The parent node in the tree, which is `None` for the root node.
* The reasoning step that represents the specific node.
* The contextual memory, which represents the context up to the current reasoning step. The context of the root node contains the initial task as the first item of the list.
* a list of the children nodes
* A flag indicating whether the current node is a terminal node, i.e. the final step of the reasoning.
* Another flag, which indicates whether the subtree below the current node leads to any terminal nodes.
* additional attributes that directly relate to the MCTS:
  * The number of times the node has been visited.
  * reward associated with the reasoning step
  * initial prediction of the quality value
  * quality value
  * average value of the simulations
  * a list of simulations that started from this reasoning step
  * advantage of the node


### Functionality

The Node class also provides a number of methods:
* initialization
* check, whether the node was already expanded
* determination of the child node with the highest selection score
* determination of the selection scores of all child nodes
* execution of simulations starting from the current reasoning step
* generation of simulations and evaluation of the reasoning step using the evaluation transformation
* expansion of the node by generating child nodes with the next reasoning steps and evaluation of these child nodes (supports concurrent execution)
* update the quality value of the node and recursively backpropagate that quality value to the parent node
* two debug/documentation methods: creation of a dictionary representation of a node and storage of the whole tree in a file

## Reasoning Strategy: Monte Carlo Tree Search Class

The reasoning strategy is implemented as a class called MCTS, that uses the [expansion](../expansion/README.md) and [evaluation](../evaluation/README.md) components for the generate and evaluate (of intermediate and final reasoning steps) transformations.
Additionally the search can be configured by using the [args](#configuration) argument.
The implementation of the MCTS class can be found in the file [mcts.py](mcts.py##L437).

### Functionality

The MCTS class provides the following methods:
* initialization
* execution of the Monte Carlo Tree Search
* selection of the best path based on a criteria
* selection of the best path based on the quality value
* tree traversal in a DFS manner

## Configuration

The `args` dictionary is used to provide the general configuration for both the Node class as well as the MCTS class.

`args` supports the following keys:
```
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
```
