from pathlib import Path
from typing import Callable

import pandas as pd
from natsort import natsorted  # sorts lists "naturally"

from YACS.yacs import Config
from src.utils.reproducibility import load_configuration, path_to_project_root


def iterate_over_semester_data(
    subdirectory: str,
    function_to_perform: Callable[[Path, str, Config], None],
) -> None:
    """
    Iterate over the different semesters of training data and generate the training data with action
    and reward columns.

    Args:
        subdirectory: The subdirectory of the data folder to iterate over.
        function_to_perform: The function to perform on each semester folder.

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
        # iterate over the different exercises of training data
        exercise_file_path_generator = natsorted(  # sort the files naturally
            # natsorted was chosen to sort the files naturally because the default sort
            # function sorts the files lexicographically, which is not what we want
            (  # ignore any problem-level data in this subdirectory
                path_to_project_root() / "data" / "with_inferred_rewards"
            ).glob(f"{problem_id}*_*.csv")
        )  # find all the .csv files in the directory that have the pattern "*_*.csv"

        if len(exercise_file_path_generator) == 0:
            print(f"No data found for {problem_id}...")
            continue
        file = exercise_file_path_generator[-1]  # get the most recent data for the exercise
        if file.is_dir():
            print(f"Skipping {file.name} (it is a directory)...")
            continue

        print(
            f"Selecting the most recent data for {file.name}..."
        )

        # get the most up-to-date data for the exercise
        data_frame = pd.read_csv(file)

        # create a subdirectory to store the data for policy induction
        output_directory = path_to_project_root() / "data" / "for_policy_induction"
        output_directory.mkdir(parents=True, exist_ok=True)
        # save the data frame to a .csv file
        data_frame.to_csv(
            output_directory / f"{problem_id}.csv",
            index=False,
        )


if __name__ == "__main__":
    iterate_over_semester_data(
        subdirectory="with_inferred_rewards",
        function_to_perform=None
    )