"""
This module implements functions to look up the grades for the students in the given semester and
append them to the grades data if they are missing. This is necessary because the grades data
does not contain the grades for all students in the semester, so we need to look up the grades
for the students that are missing.
"""
from pathlib import Path

import pandas as pd

from YACS.yacs import Config
from src.preprocess.data.to_infernet_format import make_year_and_semester_int


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
    if config.data.grades.lookup_missing and semester_name in config.data.grades.append_from:
        summary_df = pd.read_csv(semester_folder / "Summary" / "experiment_summary.csv")
        semester_grades_df = pd.read_csv(
            semester_folder / "Summary" / "user_final_grades.csv"
        )[["userID", "questionID", "overallScoreNew"]]
        # semester_folder.parent is the folder containing the semester folders (e.g., "data/raw")
        path_to_grades_directory = semester_folder.parent / "Scores"
        grades_df = pd.read_csv(
            path_to_grades_directory / f"{config.data.grades.name}.csv"
        )

        year_int, semester_int = make_year_and_semester_int(semester_name)
        # if the minimum user ID is less than the minimum user ID for the semester, then add the
        # year_int and semester_int to the user IDs
        if summary_df["userID"].min() < year_int + semester_int:
            summary_df["userID"] = summary_df["userID"] + year_int + semester_int

        if semester_grades_df["userID"].min() < year_int + semester_int:
            semester_grades_df["userID"] = (
                semester_grades_df["userID"] + year_int + semester_int
            )

        if not summary_df["userID"].isin(grades_df["userID"]).any():
            user_scores = []
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
                    user_grade: float = summary_df[summary_df["userID"] == user_id][
                        "NLG_post_SR"
                    ].values[0]
                    user_scores_df[config.data.grades.metric] = user_grade
                    # append the user scores to the list of user scores
                    user_scores.append(user_scores_df)
            # append the user scores to the grades data frame
            grades_df = pd.concat([grades_df, pd.concat(user_scores)])
            # drop users with NaN score and save the grades data frame
            grades_df[grades_df[config.data.grades.metric].notna()].to_csv(
                path_to_grades_directory / f"{config.data.grades.name}.csv", index=False
            )
