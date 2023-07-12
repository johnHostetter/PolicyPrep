"""
Use the DQN algorithm from d3rlpy to train a policy.
"""
import multiprocessing as mp

import pandas as pd
from sklearn.model_selection import train_test_split

from d3rlpy.algos import DQN
from d3rlpy.dataset import MDPDataset
from d3rlpy.metrics import td_error_scorer, average_value_estimation_scorer

from src.utils.reproducibility import load_configuration, path_to_project_root


def induce_dqn_policies(num_workers: int = 1) -> None:
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
                train_dqn,
                args=(
                    mdp_dataset,
                    problem_id,
                ),
            )
        pool.close()
        pool.join()


def train_dqn(mdp_dataset: MDPDataset, problem_id: str) -> None:
    """
    Train a DQN policy.

    Args:
        mdp_dataset: The MDP dataset for the specified problem ID.
        problem_id: The ID of the problem, or "problem" if the dataset is "problem-level".

    Returns:
        None
    """
    print(f"Training a DQN policy for {problem_id}...")
    dqn = DQN(use_gpu=False)

    train_episodes, test_episodes = train_test_split(mdp_dataset, test_size=0.2)
    dqn.fit(
        train_episodes,
        eval_episodes=test_episodes,
        n_epochs=10,
        scorers={
            "td_error": td_error_scorer,
            "value_scale": average_value_estimation_scorer,
        },
    )
    print(f"Finished training a DQN policy for {problem_id}...")
    # make the directory if it does not exist already and save the model
    print(f"Saving the DQN policy for {problem_id}...")
    output_directory = path_to_project_root() / "models" / "d3rlpy"
    output_directory.mkdir(parents=True, exist_ok=True)
    dqn.save_model(
        str(path_to_project_root() / "models" / "d3rlpy" / f"{problem_id}.pt")
    )
    print(f"Finished saving the DQN policy for {problem_id}...")


if __name__ == "__main__":
    induce_dqn_policies()
