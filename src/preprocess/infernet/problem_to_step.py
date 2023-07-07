"""
This script is used to propagate the rewards from the problem level to the step level.
"""
import multiprocessing as mp
from collections import defaultdict
from pathlib import Path
from typing import Dict

import pandas as pd

from YACS.yacs import Config
from src.preprocess.data.selection import get_most_recent_data
from src.preprocess.infernet.common import read_data
from src.utils.reproducibility import load_configuration, path_to_project_root


def propagate_problem_level_rewards_to_step_level():
    """
    Propagate the rewards from the problem level to the step level.
    """
    # load the configuration file
    config = load_configuration()
    # get the most recent problem-level data (with inferred rewards)
    path_to_most_recent_data = get_most_recent_data(problem_id="problem")
    problem_data = pd.read_csv(path_to_most_recent_data, header=0)

    # get the list of users in the problem-level data
    user_list = problem_data["userID"].unique()
    user_ids = problem_data.userID.values
    problems = problem_data.problem.values
    infer_rewards = problem_data.inferred_reward.values

    # create a dictionary of the form {user_id: {problem: reward}}
    user_problem_reward = defaultdict(dict)
    for iteration, user_id in enumerate(user_ids):
        problem = problems[iteration]
        # problem ID contains "w" if the exercise is shown as a word problem
        if "w" in problem[-1]:  # if the last character is "w", remove it
            problem = problem[:-1]  # remove the last character
        infer_reward = infer_rewards[iteration]
        user_problem_reward[user_id][problem] = infer_reward

    # iterate over the different exercises of training data
    exercise_file_path_generator = (
        (  # ignore any problem-level data in this subdirectory
            path_to_project_root() / "data" / "for_propagating_rewards"
        ).glob("*(w).csv")
    )  # find all the .csv files in the directory that end with "(w).csv"

    num_workers = mp.cpu_count() - 1
    with mp.Pool(processes=num_workers) as pool:
        for file in exercise_file_path_generator:
            if file.is_dir():
                print(f"Skipping {file.name} (it is a directory)...")
                continue

            print(
                f"Propagating inferred immediate rewards from problem-level to {file.name}..."
            )
            step_data = read_data(
                file.name, subdirectory="for_propagating_rewards", selected_users=user_list
            )

            pool.apply_async(
                propagate_problem_reward_to_step_level_data,
                args=(
                    config,
                    file,
                    step_data,
                    user_problem_reward,
                ),
            )
        pool.close()
        pool.join()

    print(
        "Problem-level rewards have been propagated to all step-level data. "
        "Training of step-level InferNet models can now begin."
    )


def propagate_problem_reward_to_step_level_data(
    config: Config,
    file: Path,
    step_data: pd.DataFrame,
    user_problem_reward: Dict[str, Dict[str, float]],
) -> None:
    """
    Propagate the rewards from the problem level to the step level. This function is
    called by the `propagate_problem_level_rewards_to_step_level` function. It is
    called in parallel for each exercise. The function modifies the `step_data`
    DataFrame in-place. The `step_data` DataFrame is saved to the appropriate
    subdirectory.

    Args:
        config: The configuration file.
        file: The path to the step-level data.
        step_data: The step-level data.
        user_problem_reward: A dictionary of the form {user_id: {problem: reward}}.

    Returns:
        None
    """
    user_ids = step_data["userID"].unique()
    for user in user_ids:
        if (
            # skip users that are not in the problem-level data
            user in config.training.skip.users
            # skip users that have not solved all the problems
            or len(user_problem_reward[user].keys()) != len(config.training.problems)
        ):
            continue
        for problem in config.training.problems:
            if problem in config.training.skip.problems:
                continue  # skip the problem; it is not used for training
            nn_inferred_reward = user_problem_reward[user][problem]
            step_data.loc[
                (step_data.userID == user)
                & (step_data.problem == problem)
                & (step_data.decisionPoint == "probEnd"),
                "reward",
            ] = nn_inferred_reward
    step_data.to_csv(
        path_to_project_root() / "data" / "for_inferring_rewards" / file.name,
        index=False,
    )


if __name__ == "__main__":
    propagate_problem_level_rewards_to_step_level()
