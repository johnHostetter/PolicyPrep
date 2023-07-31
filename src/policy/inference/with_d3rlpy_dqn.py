"""
This script contains the code necessary to perform policy inference using D3RLPy. This script will
produce the Q-values for each action for each state in the dataset. The Q-values are saved to a
.csv file and may be used to facilitate policy evaluation.
"""

import torch
import d3rlpy.algos

import pandas as pd
from d3rlpy.dataset import MDPDataset

from YACS.yacs import Config
from src.utils.reproducibility import load_configuration, path_to_project_root


def calculate_d3rlpy_algo_q_values(config: Config = None) -> None:
    """
    Calculate the Q-values for each action for each state in the dataset. The Q-values are saved
    to a .csv file and may be used to facilitate policy evaluation. This function will calculate
    the Q-values for each action for each state in the dataset for each problem in the training
    set. The Q-values are saved to a .csv file and may be used to facilitate policy evaluation.

    Args:
        config: The configuration settings.

    Returns:
        None
    """
    if config is None:
        config = load_configuration("default_configuration.yaml")
    problems = ["problem"]
    problems.extend(list(config.training.problems))
    for algo in config.training.algorithms:
        for problem in problems:
            if problem not in config.training.skip.problems:
                if "problem" in problem:
                    state_features = config.data.features.problem
                    possible_actions = config.training.actions.problem
                else:
                    problem += "(w)"
                    state_features = config.data.features.step
                    possible_actions = config.training.actions.step

                # load the dataset
                path_to_d3rlpy_data = (   # we need the d3rlpy dataset to build with the model
                    path_to_project_root() / "data" / "for_policy_induction" / "d3rlpy" / f"{problem}.h5"
                )
                path_to_pandas_data = (
                    path_to_project_root() / "data" / "for_policy_induction" / "pandas" / f"{problem}.csv"
                )
                dataset = pd.read_csv(str(path_to_pandas_data))
                mdp_dataset = MDPDataset.load(str(path_to_d3rlpy_data))
                # load the model and retrieve the implementation of the Q-function
                path_to_model = path_to_project_root() / "models" / "d3rlpy" / f"{problem}.pt"
                alg = d3rlpy.algos.get_algo(algo,discrete=bool)

                d3rlpy_algorithm = alg()
                d3rlpy_algorithm.build_with_dataset(mdp_dataset)
                d3rlpy_algorithm.load_model(str(path_to_model))
                q_function_implementation = d3rlpy_algorithm._impl.q_function._q_funcs[0]

                # compute the Q-values for each state-action pair
                q_values = q_function_implementation(
                    torch.tensor(dataset[list(state_features)].values).float()
                ).detach().numpy()
                # TODO: check that the Q-values are correct for each action
                q_values_df = pd.DataFrame(
                    q_values, columns=[f"{action}_Q_value" for action in possible_actions]
                )
                results_df = pd.concat([dataset, q_values_df], axis=1)
                path_to_output_directory = (
                    path_to_project_root() / "data" / "for_policy_evaluation" / algo
                )
                path_to_output_directory.mkdir(parents=True, exist_ok=True)
                results_df.to_csv(
                    path_to_output_directory / f"{problem}.csv",
                    index=False,
                )

if __name__ == "__main__":
    calculate_d3rlpy_algo_q_values()
