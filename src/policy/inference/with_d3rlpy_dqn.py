"""
This script contains the code necessary to perform policy inference using D3RLPy. This script will
produce the Q-values for each action for each state in the dataset. The Q-values are saved to a
.csv file and may be used to facilitate policy evaluation.
"""
import torch
import d3rlpy.algos

import pandas as pd
from alive_progress import alive_bar
from d3rlpy.dataset import MDPDataset
from colorama import (
    init as colorama_init,
)  # for cross-platform colored text in the terminal
from colorama import Fore, Style  # for cross-platform colored text in the terminal

from YACS.yacs import Config
from src.utilities.reproducibility import load_configuration, path_to_project_root

colorama_init(autoreset=True)  # for cross-platform colored text in the terminal


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
    for algorithm_str in config.training.algorithms:
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
                path_to_d3rlpy_data = (  # we need the d3rlpy dataset to build with the model
                    path_to_project_root()
                    / "data"
                    / "for_policy_induction"
                    / config.training.data.policy
                    / "d3rlpy"
                    / f"{problem}.h5"
                )
                path_to_pandas_data = (
                    path_to_project_root()
                    / "data"
                    / "for_policy_induction"
                    / config.training.data.policy
                    / "pandas"
                    / f"{problem}.csv"
                )
                dataset = pd.read_csv(str(path_to_pandas_data))
                mdp_dataset = MDPDataset.load(str(path_to_d3rlpy_data))
                # load the model and retrieve the implementation of the Q-function
                alg = d3rlpy.algos.get_algo(algorithm_str, discrete=True)
                path_to_model = (
                    path_to_project_root()
                    / "models"
                    / "policies"
                    / config.training.data.policy
                    / alg.__name__
                    / "onnx"
                    / f"{problem}.onnx"
                )
                import onnxruntime as ort

                q_function_implementation = ort.InferenceSession(path_to_model)
                # q_function_implementation = torch.jit.load(path_to_model)

                # compute the Q-values for each state-action pair
                # path_to_norms = (
                #     path_to_project_root()
                #     / "data/normalization_values/{}.csv".format(problem)
                # )
                import numpy as np

                # normalization_df = pd.read_csv(path_to_norms)
                # min_vectors = normalization_df.min_val.values.astype(np.float64)
                # max_vectors = normalization_df.max_val.values.astype(np.float64)
                # filter = max_vectors != min_vectors
                filter = np.ones(len(state_features), dtype=bool)
                from torch.utils.data import DataLoader

                sample_data = torch.tensor(
                    np.take(
                        dataset[list(state_features)].values,
                        np.where(filter)[0],
                        axis=1,
                    )
                ).float()
                print(
                    f"{Fore.YELLOW}"
                    f"Policy: {algorithm_str.upper()} | "
                    f"Data: {config.training.data.policy.capitalize()} {tuple(sample_data.shape)}"
                    f"{Style.RESET_ALL}"
                )
                dataloader = DataLoader(sample_data, batch_size=1)
                q_values = []
                with alive_bar(len(dataloader), title=f"ID: {problem}") as bar:
                    for batch in dataloader:
                        q_values.append(
                            q_function_implementation.run(
                                None, {"x": batch.detach().numpy().astype(np.float32)}
                            )[0]
                            # .cpu()
                            # .detach()
                            # .numpy()
                        )
                        bar()
                    q_values = np.squeeze(np.array(q_values), axis=1)
                    assert q_values.shape[-1] == len(possible_actions)  # sanity check

                    print(
                        f"Action Distribution: {np.unique(q_values.argmax(axis=-1), return_counts=True)}"
                    )

                    # TODO: check that the Q-values are correct for each action
                    q_values_df = pd.DataFrame(
                        q_values,
                        columns=[f"{action}_Q_value" for action in possible_actions],
                    )
                    results_df = pd.concat([dataset, q_values_df], axis=1)
                    path_to_output_directory = (
                        path_to_project_root()
                        / "data"
                        / "for_policy_evaluation"
                        / config.training.data.policy
                        / algorithm_str
                    )
                    path_to_output_directory.mkdir(parents=True, exist_ok=True)
                    results_df.to_csv(
                        path_to_output_directory / f"{problem}.csv",
                        index=False,
                    )


if __name__ == "__main__":
    calculate_d3rlpy_algo_q_values()
