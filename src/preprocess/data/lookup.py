"""
This module implements functions to look up the grades for the students in the given semester and
append them to the grades data if they are missing. This is necessary because the grades data
does not contain the grades for all students in the semester, so we need to look up the grades
for the students that are missing.
"""
import warnings
from pathlib import Path
from typing import List

import pandas as pd
from colorama import (
    init as colorama_init,
)  # for cross-platform colored text in the terminal
from colorama import Fore, Style  # for cross-platform colored text in the terminal

from YACS.yacs import Config
from src.preprocess.data.to_infernet_format import make_year_and_semester_int

colorama_init()  # initialize colorama


def lookup_semester_grades_and_append_if_missing(
    semester_folder: Path, semester_name: str, config: Config
) -> None:
    """
    Lookup the grades for the students in the given semester and append them to the grades data
    if they are missing. It is necessary to do this because the grades data does not contain the
    grades for all students in the semester, so we need to look up the grades for the students
    that are missing. It will only do this if the semester is in the list of semesters to append
    from in the configuration settings. This function will also add the year and semester to the
    user IDs if they are not already there. This is necessary because the user IDs are not unique
    across semesters, so we need to add the year and semester to the user IDs to make them unique.

    It will also remove the offset used for wording problems. This is necessary because
    the offset is not consistent across semesters, so we need to remove it to make the problem IDs
    consistent across semesters.

    Lastly, it will remove any users that have NaN grades. This is necessary because the grades
    data contains users that have NaN grades, and subsequent steps in the pipeline may not properly
    check for this scenario.

    Args:
        semester_folder: The folder containing the semester data.
        semester_name: The name of the semester.
        config: The configuration settings.

    Returns:
        None
    """
    if (
        config.data.grades.lookup_missing
        and semester_name in config.data.grades.append_from
    ):
        semester_summary_df: pd.DataFrame = pd.read_csv(
            semester_folder / "Summary" / "experiment_summary.csv"
        )
        semester_grades_df: pd.DataFrame = pd.read_csv(
            semester_folder / "Summary" / "user_final_grades.csv"
        )[["userID", "questionID", "overallScoreNew"]]
        # semester_folder.parent is the folder containing the semester folders (e.g., "data/clean")
        path_to_grades_directory: Path = semester_folder.parent / "Scores"
        grades_df: pd.DataFrame = pd.read_csv(
            path_to_grades_directory / f"{config.data.grades.name}.csv"
        )

        # the user IDs are not unique across semesters, so we need to add the year and semester
        # this should have already been done, but just in case it hasn't, we will check here
        year_int, semester_int = make_year_and_semester_int(semester_name)
        assert (
            semester_summary_df["userID"].min() >= year_int + semester_int
        ), "The year and semester have not been appended to the user IDs yet."
        assert (
            semester_grades_df["userID"].min() >= year_int + semester_int
        ), "The year and semester have not been appended to the user IDs yet."

        if not semester_summary_df["userID"].isin(grades_df["userID"]).any():
            user_scores: List[pd.DataFrame] = []
            for user_id, group in semester_grades_df.groupby("userID"):
                if user_id not in grades_df["userID"].unique():
                    user_scores_df = (
                        group.pivot(columns="questionID")["overallScoreNew"]
                        .max()
                        .to_frame()
                        .transpose()
                    )
                    # remove the offset used for wording problems
                    # and rename the columns to be strings
                    question_id_offset = 1000
                    user_scores_df.columns = [
                        str(column - question_id_offset)
                        if column > question_id_offset
                        else str(column)
                        for column in user_scores_df.columns
                    ]
                    # convert column types from int to string
                    # user_scores_df.columns = user_scores_df.columns.astype(str)
                    # add the userID column
                    user_scores_df.insert(0, "userID", user_id)
                    # find the missing columns and add them
                    missing_columns = set(grades_df.columns) - set(
                        user_scores_df.columns
                    )
                    for column in missing_columns:
                        user_scores_df[column] = None
                    # reorder the columns
                    user_scores_df = user_scores_df[grades_df.columns]
                    # update the score metric we are interested in
                    user_grade: float = semester_summary_df[
                        semester_summary_df["userID"] == user_id
                    ]["NLG_post_SR"].values[0]
                    user_scores_df[config.data.grades.metric] = user_grade
                    # append the user scores to the list of user scores
                    user_scores.append(user_scores_df)
            # append the user scores to the grades data frame
            new_user_scores_df: pd.DataFrame = pd.concat(user_scores)
            print(
                f"{Fore.GREEN}"
                f"{semester_name}: Appending {len(new_user_scores_df)} "
                f"users to the grades data frame..."
                f"{Style.RESET_ALL}"
            )
            grades_df = pd.concat([grades_df, new_user_scores_df])
            # drop users with NaN score and save the grades data frame
            num_of_users_with_nan_scores = grades_df[
                grades_df[config.data.grades.metric].isna()
            ].shape[0]
            if num_of_users_with_nan_scores > 0:
                warnings.warn(
                    f"{Fore.RED}"
                    f"Dropping {num_of_users_with_nan_scores} "
                    f"users with NaN grades from the grades data frame..."
                    f"{Style.RESET_ALL}"
                )
            grades_df[grades_df[config.data.grades.metric].notna()].to_csv(
                path_to_grades_directory / f"{config.data.grades.name}.csv", index=False
            )
        else:
            pass
