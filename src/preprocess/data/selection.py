"""
This module contains the code to select the most recent data for each exercise and problem
for policy induction. The data is saved to a subdirectory of the data folder called
"for_policy_induction". The data is saved in two formats: (1) a .csv file and (2) a .h5 file.
The .csv file is used to train the policy by using the Pandas library for data handling. The .h5
file is used to train the policy using the D3RLPy library.
"""
from pathlib import Path

import pandas as pd
from natsort import natsorted  # sorts lists "naturally"

from src.preprocess.data.parser import data_frame_to_d3rlpy_dataset
from src.utils.reproducibility import load_configuration, path_to_project_root


def select_training_data_for_policy_induction() -> None:
    """
    Iterate over the different semesters of training data and generate the training data with action
    and reward columns.

    Returns:
        None
    """
    # load the configuration settings
    pd.options.mode.chained_assignment = None
    config = load_configuration("default_configuration.yaml")

    problems = list(config.training.problems)
    problems.insert(0, "problem")

    for problem_id in problems:
        if "problem" not in problem_id:
            problem_id += "(w)"
        file = get_most_recent_data(problem_id)

        print(f"Selecting the most recent data from {file.name}...")

        # get the most up-to-date data for the exercise
        data_frame = pd.read_csv(file)

        # create subdirectories to store the data for policy induction
        output_directory = path_to_project_root() / "data" / "for_policy_induction"
        for subdirectory in ["pandas", "d3rlpy"]:
            (output_directory / subdirectory).mkdir(parents=True, exist_ok=True)

        # save the data frame to a .csv file in the subdirectory "pandas" for policy induction
        data_frame.to_csv(
            output_directory / "pandas" / f"{problem_id}.csv",
            index=False,
        )
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


def get_most_recent_data(problem_id) -> Path:
    """
    Get the path to the most recent data for the given exercise.

    Args:
        problem_id: The problem ID of the exercise, or "problem" if the data is for the
        problem-level.

    Returns:
        The path to the most recent data for the given exercise.
    """
    # iterate over the different exercises of training data
    exercise_file_path_generator = natsorted(  # sort the files naturally
        # natsorted was chosen to sort the files naturally because the default sort
        # function sorts the files lexicographically, which is not what we want
        (  # ignore any problem-level data in this subdirectory
                path_to_project_root() / "data" / "with_inferred_rewards"
        ).glob(f"{problem_id}*_*.csv")
    )  # find all the .csv files in the directory that have the pattern "*_*.csv"
    if len(exercise_file_path_generator) == 0:
        raise FileNotFoundError(f"No data found for {problem_id}...")
    file = exercise_file_path_generator[
        -1
    ]  # get the most recent data for the exercise
    if file.is_dir():
        raise FileNotFoundError(f"Skipping {file.name} (it is a directory)...")
    return file


if __name__ == "__main__":
    select_training_data_for_policy_induction()
