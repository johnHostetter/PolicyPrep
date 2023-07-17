"""
This module contains the code to select the most recent data for each exercise and problem
for policy induction. The data is saved to a subdirectory of the data folder called
"for_policy_induction". The data is saved in two formats: (1) a .csv file and (2) a .h5 file.
The .csv file is used to train the policy by using the Pandas library for data handling. The .h5
file is used to train the policy using the D3RLPy library.
"""
from pathlib import Path
import multiprocessing as mp
from typing import Union

import pandas as pd
from natsort import natsorted  # sorts lists "naturally"

from src.preprocess.data.parser import data_frame_to_d3rlpy_dataset
from src.utils.reproducibility import load_configuration, path_to_project_root


def select_training_data_for_policy_induction(num_workers: int = 1) -> None:
    """
    Iterate over the different semesters of training data and generate the training data with action
    and reward columns.

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
        for problem_id in problems:
            if problem_id in config.training.skip.problems:
                continue
            if "problem" not in problem_id:
                problem_id += "(w)"
            try:
                file = get_most_recent_file(
                    path_to_folder="data/with_inferred_rewards",
                    problem_id=problem_id,
                    extension=".csv",
                )
            except FileNotFoundError as file_not_found_error:
                print(repr(file_not_found_error))
                continue

            pool.apply_async(
                move_and_convert_data,
                args=(
                    file,
                    problem_id,
                ),
            )
        pool.close()
        pool.join()


def move_and_convert_data(file: Path, problem_id: str) -> None:
    """
    Move the data to a subdirectory of the data folder called "for_policy_induction" and convert
    the data frame to an MDP dataset.

    Args:
        file: the path to the .csv file containing the data
        problem_id: the ID of the problem

    Returns:
        None
    """
    print(f"Selecting the most recent data from {file.name}...")
    # get the most up-to-date data for the exercise
    data_frame = pd.read_csv(file)
    # create subdirectories to store the data for policy induction
    output_directory = path_to_project_root() / "data" / "for_policy_induction"
    for subdirectory in ["pandas", "d3rlpy"]:
        (output_directory / subdirectory).mkdir(parents=True, exist_ok=True)
    # before converting the data frame to an MDP dataset, we need to create a new column
    # called "action" that contains the action taken by the tutor
    # this is necessary because the data frame contains three columns that contain the
    # called "action_ps", "action_fwe", and "action_we" that contain the action taken by
    # the tutor and is essentially a one-hot encoding of the action taken by the tutor
    if "problem" in problem_id:
        action_columns = ["action_ps", "action_fwe", "action_we"]
    else:
        action_columns = ["action_ps", "action_we"]
    action_encoding = range(len(action_columns))
    # the following code creates a new column called "action" that contains the action
    # taken by the tutor and reverses this one-hot encoding by choosing the column label for
    # each row where the label has the maximum value
    data_frame["action"] = data_frame[action_columns].idxmax(1)

    # save the data frame to a .csv file in the subdirectory "pandas" for policy induction
    data_frame.to_csv(
        output_directory / "pandas" / f"{problem_id}.csv",
        index=False,
    )

    # drop the columns that contain the one-hot encoding of the action taken by the tutor
    data_frame.drop(
        columns=action_columns,
        inplace=True,
    )
    # remove the "action_" prefix from the action column
    data_frame["action"] = data_frame["action"].str.replace("action_", "")
    action_columns = [action.replace("action_", "") for action in action_columns]
    # convert the action column to an integer:
    # if problem-level: problem (ps) -> 0, step_decision (fwe) -> 1, example (we) -> 2
    # if exercise-level: problem (ps) -> 0, example (we) -> 1
    data_frame["action"] = data_frame["action"].replace(
        to_replace=action_columns, value=action_encoding
    )
    # convert the data frame to a Markov Decision Process (MDP) dataset
    mdp_dataset = data_frame_to_d3rlpy_dataset(
        features_df=data_frame, problem_id=problem_id
    )
    # save the MDP dataset to a .h5 file (the default format for D3RLPy)
    mdp_dataset.dump(str(output_directory / "d3rlpy" / f"{problem_id}.h5"))


def get_most_recent_file(
    path_to_folder: Union[str, Path], problem_id: str, extension: str
) -> Path:
    """
    Get the path to the most recent data for the given exercise.

    Args:
        path_to_folder: The path to the directory containing the data, models, or logs.
        problem_id: The problem ID of the exercise, or "problem" if the data is for the
        problem-level.
        extension: The type of file to get the most recent file for. This can be ".csv", ".h5",
        or ".pth".

    Returns:
        The path to the most recent data for the given exercise.
    """
    if not isinstance(path_to_folder, Path):
        path_to_folder = Path(path_to_folder)

    # iterate over the different exercises of training data
    exercise_file_path_generator = natsorted(  # sort the files naturally
        # natsorted was chosen to sort the files naturally because the default sort
        # function sorts the files lexicographically, which is not what we want
        (  # ignore any problem-level data in this subdirectory
            path_to_project_root() / path_to_folder
        ).glob(f"{problem_id}*_*{extension}")
    )  # find all the .csv files in the directory that have the pattern "*_*.csv"
    if len(exercise_file_path_generator) == 0:
        raise FileNotFoundError(f"No files found for {problem_id}...")
    file = exercise_file_path_generator[-1]  # get the most recent data for the exercise
    if file.is_dir():
        raise FileNotFoundError(f"Skipping {file.name} (it is a directory)...")
    return file


if __name__ == "__main__":
    select_training_data_for_policy_induction()
