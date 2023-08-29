"""
Use the DQN algorithm from d3rlpy to train a policy.
"""
import multiprocessing as mp

import torch
import d3rlpy.algos
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from d3rlpy.dataset import MDPDataset
from d3rlpy.metrics import td_error_scorer, average_value_estimation_scorer

from src.utils.reproducibility import load_configuration, path_to_project_root


def induce_policies_with_d3rlpy(num_workers: int = 1) -> None:
    """
    Iterate over the different semesters of training data and induce policies using the DQN
    algorithm.

    Args:
        num_workers: The number of processes to use for multiprocessing.

    Returns:
        None
    """
    # load the configuration settings
    pd.options.mode.chained_assignment = None
    config = load_configuration("default_configuration.yaml")

    problems = list(config.training.problems)
    problems.insert(0, "problem")

    with mp.Pool(processes=num_workers) as pool:
        path_to_policy_data_directory = (
            path_to_project_root() / "data" / "for_policy_induction"
        )
        for algo in config.training.algorithms:
            for problem_id in problems:
                if problem_id in config.training.skip.problems:
                    continue
                if "problem" not in problem_id:
                    problem_id += "(w)"
                try:
                    path_to_data = (
                        path_to_policy_data_directory / "d3rlpy" / f"{problem_id}.h5"
                    )
                    if not path_to_data.exists():
                        print(f"Skipping {path_to_data.name} (it does not exist)...")
                        continue
                    if path_to_data.is_dir():
                        print(f"Skipping {path_to_data.name} (it is a directory)...")
                        continue
                    print(f"Using data from {path_to_data.name}...")
                    mdp_dataset = MDPDataset.load(str(path_to_data))

                except FileNotFoundError as file_not_found_error:
                    print(repr(file_not_found_error))
                    continue

                pool.apply_async(
                    train_d3rlpy_policy,
                    args=[
                        mdp_dataset,
                        problem_id,
                        d3rlpy.algos.get_algo(algo, discrete=True),
                    ],
                )
        pool.close()
        pool.join()


def train_d3rlpy_policy(mdp_dataset: MDPDataset, problem_id: str, d3rlpy_alg) -> None:
    """
    Train a DQN policy.

    Args:
        mdp_dataset: The MDP dataset for the specified problem ID.
        problem_id: The ID of the problem, or "problem" if the dataset is "problem-level".

    Returns:
        None
    """
    print(f"Training a {d3rlpy_alg.__name__} policy for {problem_id}...")
    algorithm = d3rlpy_alg(use_gpu=False)

    train_episodes, test_episodes = train_test_split(mdp_dataset, test_size=0.2)
    algorithm.fit(
        train_episodes,
        eval_episodes=test_episodes,
        n_epochs=10,
        scorers={
            "td_error": td_error_scorer,
            "value_scale": average_value_estimation_scorer,
        },
    )
    print(f"Finished training a {d3rlpy_alg.__name__} policy for {problem_id}...")
    # make the directory if it does not exist already and save the model
    print(f"Saving the {d3rlpy_alg.__name__} policy for {problem_id}...")
    for directory in ["d3rlpy", "trace", "onnx"]:
        output_directory = path_to_project_root() / "models" / "policies"
        output_directory = output_directory / d3rlpy_alg.__name__ / directory
        output_directory.mkdir(parents=True, exist_ok=True)
        if directory == "d3rlpy":
            # save the algorithm model to the output directory
            algorithm.save_model(str(output_directory / f"{problem_id}.pt"))
        elif directory == "trace":
            # # save the traced model to the output directory (for use in the web app)
            # max_length = max([len(episode.observations) for episode in train_episodes])
            # train_observations = torch.tensor(np.array([
            #     torch.nn.functional.pad(
            #         torch.tensor(episode.observations),
            #         pad=(0, 0, 0, max_length - episode.observations.shape[0])
            #     ).numpy() for episode in train_episodes
            # ]))
            # train_observations = train_observations.view(-1, train_observations.shape[-1]).cpu()
            # # fetch the function approximator that is used for the Q-function
            # model = algorithm.impl.q_function._q_funcs[0]._encoder
            # # enable the evaluation mode for the model to be traced
            # model.eval()  # torch.nn.Parameters cannot be traced in training mode
            # model.cpu()  # the traced model will be used in the web app, which is CPU-only
            # traced_model = torch.jit.trace(
            #     model, train_observations
            # )
            # torch.jit.save(traced_model, str(output_directory / f"{problem_id}.pt"))

            # save greedy-policy as TorchScript
            algorithm.save_policy(str(output_directory / f"{problem_id}.pt"))
        else:
            # save greedy-policy as ONNX
            algorithm.save_policy(str(output_directory / f"{problem_id}.onnx"))
    print(f"Finished saving the {d3rlpy_alg.__name__} policy for {problem_id}...")


if __name__ == "__main__":
    induce_policies_with_d3rlpy()
