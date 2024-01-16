"""
This Python module contains the code to generate training data with action and reward columns. This
expects the data to be in the format generated by the `download_data.py` script. The data used must
be from the post-processing step; for example, features_all.csv must be from the PostProcessing
folder, not the Experiment folder. The generated format of the training data is required for the
InferNet code.
"""
import sys
import warnings
from pathlib import Path
from timeit import timeit

from alive_progress import alive_bar, alive_it
from typing import Tuple, Union, List, Callable

import numpy as np
import pandas as pd
from colorama import (
    init as colorama_init,
)  # for cross-platform colored text in the terminal
from colorama import Fore, Style  # for cross-platform colored text in the terminal

from YACS.yacs import Config
from src.utilities.reproducibility import load_configuration, path_to_project_root

colorama_init()  # initialize colorama


def make_year_and_semester_int(semester: str) -> Union[Tuple[int, int], None]:
    """
    Convert a semester string to a year and semester integer.

    Args:
        semester: A string representing a semester, e.g. 'F19'.

    Returns:
        A tuple of the year and semester integers, following our lab's naming convention. If the
        semester string is invalid, then return None.
    """
    year_int = int(semester[1:]) * 10000

    if semester[0] == "S":
        return year_int, 1000
    if semester[0] == "F":
        return year_int, 3000
    return None


def minimum_id(semester: str) -> int:
    """
    Get the minimum user ID for a given semester.

    Args:
        semester: A string representing a semester, e.g. 'F19'.

    Returns:
        The minimum user ID for the given semester.
    """
    year_int, semester_int = make_year_and_semester_int(semester)
    return year_int + semester_int + 100


def add_users_to_skip_list(user_ids: List[int], config: Config) -> None:
    """
    Append the given list of user IDs to the configuration file's list of users that
    should be skipped in later steps of the pre-processing pipeline. Modifies the configuration
    settings in-place.

    Args:
        user_ids: A list of user IDs that should be added to the skipped group.
        config: The configuration settings.

    Returns:
        The modified configuration settings.
    """
    with config.unfreeze():
        config.training.skip.users = tuple(list(config.training.skip.users) + user_ids)


def iterate_over_semester_data(
    subdirectory: str,
    function_to_perform: Callable[[Path, str, Config], None],
    config: Config,
) -> None:
    """
    Iterate over the different semesters of training data and generate the training data with action
    and reward columns.

    Args:
        subdirectory: The subdirectory of the data folder to iterate over.
        function_to_perform: The function to perform on each semester folder.
        config: The configuration file to use

    Returns:
        None
    """
    # load the configuration settings
    pd.options.mode.chained_assignment = None

    # iterate over the different semesters of training data

    semester_folder_path_generator: List[Path] = list(
        (path_to_project_root() / "data" / subdirectory).glob("* - *")
    )
    for semester_folder in semester_folder_path_generator:
        if not semester_folder.is_dir():
            print(
                f"{Fore.RED}"
                f"Skipping {semester_folder.name} (not a directory)..."
                f"{Style.RESET_ALL}"
            )
            continue
        if " - " not in semester_folder.name:
            print(
                f"{Fore.RED}"
                f"Skipping {semester_folder.name} (invalid directory name)..."
                f"{Style.RESET_ALL}"
            )
            continue

        # the name of the semester is the part of the folder name after the " - "
        # e.g. "10 - S21" -> "S21"; the part before the " - " is the folder ordering number
        (_, semester_name) = semester_folder.name.split(" - ")
        print(
            f"{Fore.YELLOW}"
            f"Processing data for the {semester_name} semester..."
            f"{Style.RESET_ALL}"
        )

        try:
            function_to_perform(semester_folder, semester_name, config)
        except FileNotFoundError as file_not_found_error:
            warnings.warn(
                f"Skipping {semester_folder.name} (no {file_not_found_error.filename} file "
                f"found for this semester)..."
            )
            continue


def convert_data_format(
    semester_folder: Path,
    semester_name: str,
    config: Config,
) -> None:
    """
    Convert the data to the format required by the InferNet training scripts.

    Args:
        semester_folder: The path to the folder containing the semester data.
        semester_name: The name of the semester, e.g. "S21".
        config: The configuration settings.

    Returns:
        None
    """
    year_int, semester_int = make_year_and_semester_int(semester_name)

    # load the grades data
    path_to_grades_directory = path_to_project_root() / "data" / "clean" / "Scores"
    try:
        grades_df = pd.read_csv(
            str(path_to_grades_directory / f"{config.data.grades.name}.csv"),
            header=0,
        )
    except FileNotFoundError:
        raise UserWarning("The pipeline cannot proceed (no grades file found).")

    # get the substep info dataframe
    substep_info_df: pd.DataFrame = get_substep_info_df(
        semester_folder, year_int, semester_int
    )
    # make the output directories for the training data
    output_directory = make_data_subdirectory(
        "with_delayed_rewards", semester_folder.name
    )
    # get the problems for the semester
    # need to prepare problem-level training data with action and reward columns
    prob_features_df: pd.DataFrame = get_features_dataframe(
        semester_folder,
        year_int,
        semester_int,
        columns=config.data.features.basic + config.data.features.problem,
    )
    # need to prepare step-level training data with action column
    step_features_df: pd.DataFrame = get_features_dataframe(
        semester_folder,
        year_int,
        semester_int,
        columns=config.data.features.basic + config.data.features.step,
    )

    # convert the data to the format required by the training scripts
    convert_problem_level_format(
        output_directory,
        semester_name,
        prob_features_df,
        substep_info_df,
        grades_df,
        config,
    )
    convert_step_level_format(
        output_directory, semester_name, step_features_df, substep_info_df, config
    )


def get_features_dataframe(
    semester_folder: Path,
    year_int: int,
    semester_int: int,
    columns: List[str],
) -> pd.DataFrame:
    """
    Get the problem-level (or step-level) features dataframes, which are used to generate the
    training data. The problem-level features dataframe contains the features for each problem
    (e.g., the number of hints requested, the number of attempts, etc.). The step-level features
    dataframe contains the features for each step (e.g., the number of hints requested, the number
    of attempts, etc.). The features dataframe generated depends on what columns are passed in. The
    columns should be a subset of the columns in the features_all.csv file.

    Args:
        semester_folder: The path to the semester folder.
        year_int: The year encoded as an integer (e.g., "23" for 2023).
        semester_int: The semester encoded as integer (e.g., "1" for Spring, "3" for Fall).
        columns: The columns to include in the dataframes. The columns should be a subset of the
            columns in the features_all.csv file.

    Returns:
        The problem-level and step-level features dataframes.
    """
    # need to prepare problem-level training data with action and reward columns
    feature_df = pd.read_csv(
        str(semester_folder / "PostProcessing" / "features_all.csv"),
        header=0,
        usecols=columns,
    )

    # update the user IDs to be unique across all semesters
    if feature_df["userID"].dtype == "object":
        feature_df["userID"] = feature_df["userID"].astype("int64")
    # if the minimum user ID is less than the minimum user ID for the semester, then add the
    # year_int and semester_int to the user IDs
    if feature_df["userID"].min() < year_int + semester_int:
        feature_df["userID"] = feature_df["userID"] + year_int + semester_int

    return feature_df


def get_problems(substep_info_df: pd.DataFrame) -> List[str]:
    """
    Get the problems for the semester (excluding the problem IDs where the tutor does not make
    a pedagogical decision, such as ex222).

    Args:
        substep_info_df: The substep info dataframe.

    Returns:
        The problems for the semester.
    """
    problems = substep_info_df["problem"].unique()
    problems = [
        problem_id
        for problem_id in problems
        if (
            problem_id[-1] != "w"
            and "ex222" not in problem_id
            and "ex144" not in problem_id
        )
    ]  # exclude word problems and ex222 and ex144 which won't be trained for step-level
    return problems


def make_data_subdirectory(subdirectory: str, semester_folder: str) -> Path:
    """
    Make a new subdirectory underneath the "data" root directory, if it does not exist already.
    Then, within that subdirectory, make a new subdirectory for the semester, if it does not exist
    already. Finally, return the path to the semester subdirectory.

    Args:
        subdirectory: The name of the subdirectory to make, if it does not exist already.
        semester_folder: The name of the semester folder to make, if it does not exist already.

    Returns:
        The path to the semester subdirectory.
    """
    output_directory = path_to_project_root() / "data" / subdirectory / semester_folder
    output_directory.mkdir(parents=True, exist_ok=True)
    return output_directory


def get_substep_info_df(
    semester_folder: Path, year_int: int, semester_int: int
) -> pd.DataFrame:
    """
    Get the substep info dataframe.

    Args:
        semester_folder: The path to the semester folder.
        year_int: The year encoded as an integer (e.g., "23" for 2023).
        semester_int: The semester encoded as integer (e.g., "1" for Spring, "3" for Fall).

    Returns:
        The substep info dataframe.
    """
    substep_info_df = pd.read_csv(
        str(semester_folder / "PostProcessing" / "substep_info.csv"), header=0
    )
    substep_info_df["userID"] = substep_info_df["userID"].astype(int)
    if substep_info_df["userID"].min() < year_int + semester_int:
        substep_info_df["userID"] = year_int + semester_int + substep_info_df["userID"]
    return substep_info_df


def convert_problem_level_format(
    output_directory: Path,
    semester_name: str,
    prob_features_df: pd.DataFrame,
    substep_info_df: pd.DataFrame,
    grades_df: pd.DataFrame,
    config: Config,
) -> None:
    """
    Convert the problem-level data to the format required by the training scripts.

    Args:
        output_directory: The path to the output directory.
        prob_features_df: The problem-level features dataframe.
        semester_name: The name of the semester, e.g. "S21".
        substep_info_df: The substep info dataframe.
        grades_df: The grades dataframe.
        config: The configuration settings.

    Returns:
        None
    """
    prob_features_df = prob_features_df[
        prob_features_df["userID"] >= minimum_id(semester_name)
    ]
    substep_info_df = substep_info_df[
        substep_info_df["userID"] >= minimum_id(semester_name)
    ]
    prob_lvl_feature_df = prob_features_df[
        prob_features_df["decisionPoint"].isin(["probStart", "probEnd"])
    ]
    prob_lvl_feature_df.drop(
        prob_lvl_feature_df[
            (prob_lvl_feature_df["decisionPoint"] == "probEnd")
            & (~prob_lvl_feature_df["problem"].isin(["ex252", "ex252w"]))
        ].index,
        inplace=True,
    )

    # eliminate duplicate row entries, with respect to the selected columns (e.g., "time")
    # for example, problem-level data from F23 has duplicate row entries
    prob_lvl_feature_df = prob_lvl_feature_df.drop_duplicates(
        ["userID", "problem", "time", "decisionPoint"]
    )

    # eliminate the data that has an insufficient number of row entries
    grouped_by_user_prob_lvl_features_df: List[pd.DataFrame] = [
        group
        for _, group in prob_lvl_feature_df.groupby(by="userID")
        if len(group)
        == (len(config.training.problems) + 1)  # the + 1 accounts for the probEnd row
    ]
    # save the users that have insufficient data associated with them
    user_ids_to_be_removed: List[int] = [
        int(user_id)
        for user_id, group in prob_lvl_feature_df.groupby(by="userID")
        if len(group)
        < (len(config.training.problems) + 1)  # the + 1 accounts for the probEnd row
    ]
    # mark these users to be skipped later
    add_users_to_skip_list(user_ids_to_be_removed, config)
    # overwrite the previous variable reference
    prob_lvl_feature_df: pd.DataFrame = pd.concat(grouped_by_user_prob_lvl_features_df)

    prob_lvl_feature_df["action"] = ""
    prob_lvl_feature_df["reward"] = ""
    users_with_nan_nlg: List[int] = []
    action_col_location = prob_lvl_feature_df.columns.get_loc("action")
    reward_col_location = prob_lvl_feature_df.columns.get_loc("reward")
    for i in alive_it(range(len(prob_lvl_feature_df)), title=f"{semester_name}"):
        user_id = prob_lvl_feature_df.iloc[i]["userID"]
        decision_point = prob_lvl_feature_df.iloc[i]["decisionPoint"]
        if decision_point == "probStart":
            problem = prob_lvl_feature_df.iloc[i]["problem"]
            unique_actions = substep_info_df[
                (substep_info_df["userID"] == user_id)
                & (substep_info_df["problem"] == problem)
            ]["substepMode"].unique()
            if len(unique_actions) == 2:
                prob_lvl_feature_df.iat[i, action_col_location] = "step_decision"
            else:
                try:
                    prob_lvl_feature_df.iat[i, action_col_location] = unique_actions[0]
                except IndexError:
                    prob_lvl_feature_df.iat[i, action_col_location] = unique_actions
        else:
            user_score: np.ndarray = grades_df[grades_df["userID"] == user_id][
                config.data.grades.metric
            ].unique()
            if len(user_score) == 0:
                continue  # the result is empty list, skip this user
            nlg = user_score[0]
            if np.isnan(nlg):
                users_with_nan_nlg.append(user_id)
            prob_lvl_feature_df.iat[i, reward_col_location] = nlg

    if len(users_with_nan_nlg):
        print(
            f"{Fore.RED}"
            f"{semester_name}: The following users have a NaN NLG score: {users_with_nan_nlg}"
            f"{Style.RESET_ALL}"  # this is not a warnings.warn message since this may be expected
        )

    # keep only the users that have a valid NLG score
    prob_lvl_feature_df = prob_lvl_feature_df[
        ~prob_lvl_feature_df["userID"].isin(users_with_nan_nlg)
    ]
    # save the problem-level training data
    prob_lvl_feature_df.to_csv(output_directory / "problem.csv", index=False)
    print(
        f"{Fore.GREEN}"
        f"{semester_name}: Saved problem-level training data to "
        f"{output_directory  / 'problem.csv'}."
        f"{Style.RESET_ALL}"
    )


def convert_step_level_format(
    output_directory: Path,
    semester_name: str,
    step_features_df: pd.DataFrame,
    substep_info_df: pd.DataFrame,
    config: Config,
) -> None:
    """
    Converts the step-level data into the format that can be used for training the step-level
    InferNet models.

    Args:
        output_directory: The path to the output directory.
        semester_name: The name of the semester.
        step_features_df: The step-level features dataframe.
        substep_info_df: The substep info dataframe.
        config: The configuration settings.

    Returns:
        None
    """
    # get the problems for the semester
    problems: List[str] = get_problems(substep_info_df)
    for problem in alive_it(problems, title=f"{semester_name}"):
        step_lvl_feature_df = step_features_df[
            (step_features_df["decisionPoint"].isin(["stepStart", "probEnd"]))
            & (step_features_df["problem"].isin([problem, problem + "w"]))
        ]
        step_lvl_feature_df["action"] = ""
        step_lvl_feature_df["reward"] = ""
        action_col_location = step_lvl_feature_df.columns.get_loc("action")
        step_lvl_substep_df = substep_info_df[
            substep_info_df["problem"].isin([problem, problem + "w"])
        ]

        # select the minimal set of user IDs to make the data agree with each other
        user_ids = set(step_lvl_feature_df.userID.unique()).intersection(
            step_lvl_substep_df.userID.unique()
        )
        # find users that have insufficient data associated with them
        user_ids_to_be_removed: List[int] = [
            int(user_id)
            for user_id, group in step_lvl_feature_df.groupby(by="userID")
            if "probEnd" not in group.decisionPoint.unique()
        ]  # e.g., user 183114 from F18
        # mark these users to be skipped later
        add_users_to_skip_list(user_ids_to_be_removed, config)
        # user IDs below 100 are considered test users - remove them, as well as other 'bad' users
        user_ids = [
            user_id
            for user_id in user_ids
            if user_id >= minimum_id(semester_name)
            and user_id not in user_ids_to_be_removed
        ]

        # make them consistent with each other
        step_lvl_feature_df = step_lvl_feature_df[
            step_lvl_feature_df.userID.isin(user_ids)
        ]
        step_lvl_substep_df = step_lvl_substep_df[
            step_lvl_substep_df.userID.isin(user_ids)
        ]

        # Feature Frame has an extra "probEnd" per problem per user.
        # Since we are at the current problem, it has an additional # of user rows
        if len(step_lvl_feature_df) != len(step_lvl_substep_df) + len(user_ids):
            new_step_lvl_substep_df: List[pd.DataFrame] = []
            new_step_lvl_feature_df: List[pd.DataFrame] = []
            print(  # this is not a warnings.warn message since this may be resolved
                f"{Fore.RED}"
                f"{semester_name}: The step-level feature and sub-step data "
                f"size mismatch for {problem}."
                f"{Style.RESET_ALL}"
            )
            # attempt to resolve the conflict/discrepancies
            for (
                user_id
            ) in (
                user_ids
            ):  # step through user-by-user, some issues can be w/ only 1 user
                # get the user's corresponding data
                tmp_user_step_lvl_substep_df = step_lvl_substep_df[
                    step_lvl_substep_df["userID"] == user_id
                ]
                tmp_user_step_lvl_feature_df = step_lvl_feature_df[
                    step_lvl_feature_df["userID"] == user_id
                ]
                # one issue with past data (e.g., S23, F23) is repeated row entries - meaning, that
                # there are multiple probEnd rows for the same problem, given a specific user
                # this is problematic as there should only be ONE
                decision_point_indices: List[int] = [
                    decision_point_index
                    for decision_point_index, decision_point in enumerate(
                        tmp_user_step_lvl_feature_df.decisionPoint.to_list()
                    )
                    if decision_point == "probEnd"
                ]  # this code goes through and finds the data indices where probEnd occurs
                # slice the user's data to only keep the first entries up until the first probEnd
                tmp_user_step_lvl_substep_df: pd.DataFrame = (
                    tmp_user_step_lvl_substep_df.iloc[0 : decision_point_indices[0]]
                )  # no + 1 needed (no probEnd row)
                tmp_user_step_lvl_feature_df: pd.DataFrame = (
                    tmp_user_step_lvl_feature_df.iloc[
                        0 : (decision_point_indices[0] + 1)
                    ]
                )  # the + 1 includes our probEnd row

                # check that the user's data is the size that we expect
                if len(tmp_user_step_lvl_substep_df) + 1 == len(
                    tmp_user_step_lvl_feature_df
                ):
                    # if it is, keep it
                    new_step_lvl_substep_df.append(tmp_user_step_lvl_substep_df)
                    new_step_lvl_feature_df.append(tmp_user_step_lvl_feature_df)
                    # print(
                    #     f"{Fore.GREEN}"
                    #     f"{semester_name}: Resolved data size mismatch for user {user_id}."
                    #     f"{Style.RESET_ALL}"
                    # )
                else:  # otherwise, we can't seem to use this user's data - something bad happened
                    # the current user has insufficient data logged (e.g., S23 user 231228)
                    print(
                        f"{Fore.RED}"
                        f"{semester_name}: Skipping user {user_id} as they have invalid data."
                        f"{Style.RESET_ALL}"
                    )  # NOTE: this isn't a common occurrence, between S23 & F23, this happened ONCE

            # override the variables
            step_lvl_substep_df: pd.DataFrame = pd.concat(objs=new_step_lvl_substep_df)
            step_lvl_feature_df: pd.DataFrame = pd.concat(objs=new_step_lvl_feature_df)
            user_ids = step_lvl_substep_df["userID"].unique().tolist()

            if len(step_lvl_feature_df) != len(step_lvl_substep_df) + len(user_ids):
                raise ValueError(
                    "Failed to resolve mismatch between feature and sub-step information. "
                    "Error is safe to ignore if this dataset can be ignored. "
                    "The error is thrown as a precaution to catch your attention!"
                )  # this error is thrown as features_info and substep_info data mismatch
                # obviously, this is rather undesirable (means there is some issue in data logging)
                # typically, this will only be thrown on newer data, which you most likely want to
                # use in your experiment study
                # you will have to dig into the data to find out what happened and where the
                # discrepancies lie, for example, semesters S23 and F23 had duplicate data rows
                # good luck! :)
            else:
                print(
                    f"{Fore.GREEN}"
                    f"{semester_name}: Mismatch and conflicts resolved between "
                    f"feature and sub-step information."
                    f"{Style.RESET_ALL}"
                )

        sub_step_counter = 0
        for i in range(len(step_lvl_feature_df)):
            decision_point = step_lvl_feature_df.iloc[i]["decisionPoint"]
            if decision_point != "stepStart":
                continue
            user_id_feature = step_lvl_feature_df.iloc[i]["userID"]
            user_id_substep = step_lvl_substep_df.iloc[sub_step_counter]["userID"]

            if user_id_feature != user_id_substep:
                print(
                    f"{Fore.RED}"
                    f"UserID mismatch between feature_all and substep_info at step-level: "
                    f"Feature -> {user_id_feature} and Substep -> {user_id_substep}"
                    f"{Style.RESET_ALL}"
                )
                continue

            step_lvl_feature_df.iat[i, action_col_location] = step_lvl_substep_df.iloc[
                sub_step_counter
            ]["substepMode"]
            sub_step_counter += 1

        # save the step-level training data for this current problem
        step_lvl_feature_df.to_csv(output_directory / f"{problem}(w).csv", index=False)
        print(
            f"{Fore.GREEN}"
            f"{semester_name}: Saved {problem} step-level training data to "
            f"{output_directory / f'{problem}(w).csv'}..."
            f"{Style.RESET_ALL}"
        )


if __name__ == "__main__":
    # example usage
    iterate_over_semester_data(
        subdirectory="clean",
        function_to_perform=convert_data_format,
        config=load_configuration("default_configuration.yaml"),
    )
