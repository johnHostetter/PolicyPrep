from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from colorama import (
    init as colorama_init,
)  # for cross-platform colored text in the terminal
from colorama import Fore, Style  # for cross-platform colored text in the terminal

from src.utilities.reproducibility import path_to_project_root

colorama_init()  # initialize colorama for cross-platform colored text in the terminal


def split_based_on_column(
    data_df: pd.DataFrame, column_name: str = "student_id"
) -> Dict[int, List[pd.DataFrame]]:
    """
    Split the dataframe based on the FSA round.

    Args:
        data_df: The dataframe to split.
        column_name: The column to split the dataframe on.

    Returns:
        The dataframe split based on the FSA round.
    """
    grouped_df = data_df.groupby(column_name)
    fsa_groups: Dict[int, List[pd.DataFrame]] = {}
    next_index: int = 0
    for _, student_df in grouped_df:  # ignore the student_id
        fsa_levels: np.ndarray = student_df["FSA_round"].unique()
        fsa_levels = fsa_levels[~np.isnan(fsa_levels)]  # drop the 'nan' value
        # find the indices in which the non-nan values are located
        for fsa_level in fsa_levels:
            prev_index: int = next_index
            next_index: int = (
                student_df[student_df["FSA_round"] == fsa_level].index[0] + 1
            )
            fsa_level_df: pd.DataFrame = data_df.iloc[prev_index:next_index]
            if fsa_level not in fsa_groups:
                fsa_groups[fsa_level] = []
            fsa_groups[fsa_level].append(fsa_level_df)
    return fsa_groups


def cprint(text: str, color: str) -> None:
    """
    Print the output in a given color.

    Args:
        text: The string to display.
    """
    print(f"{color}" f"{text}" f"{Style.RESET_ALL}")


def yprint(text: str) -> None:
    """
    Print the output in yellow.

    Args:
        text: The string to display.
    """
    cprint(text, Fore.YELLOW)


def gprint(text: str) -> None:
    """
    Print the output in yellow.

    Args:
        text: The string to display.
    """
    cprint(text, Fore.GREEN)


if __name__ == "__main__":
    data_path: Path = path_to_project_root() / "data" / "raw" / "dataset-train.csv"
    data_df: pd.DataFrame = pd.read_csv(data_path)
    yprint(
        "Begin split of the dataframe into groups first on student and then on the FSA round..."
    )
    fsa_groups: Dict[int, List[pd.DataFrame]] = split_based_on_column(data_df)
    gprint("Done")

    yprint("Merging dataframes together based on their common FSA round...")
    final_fsa_groups: Dict[int, pd.DataFrame] = {}
    for key, value in fsa_groups.items():
        all_df_for_key: pd.DataFrame = pd.concat(value, axis=0, ignore_index=False)
        # check that the number of columns are the same
        assert all_df_for_key.shape[-1] == data_df.shape[-1]
        final_fsa_groups[key] = all_df_for_key
    gprint("Done.")

    yprint("Checking our work...")
    # check that all the rows of data are accounted for
    assert (
        np.sum([df.shape[0] for df in final_fsa_groups.values()]) == data_df.shape[0]
    ), "Not all rows of data have been recovered after grouping."
    gprint(
        "Splitting the dataframe into groups dependent on the FSA round was successful."
    )

    # save the dataframes to the appropriate subdirectory
    yprint("Saving the dataframes to the appropriate subdirectory...")
    target_dir: Path = path_to_project_root() / "data" / "for_inferring_rewards"
    Path.mkdir(target_dir, exist_ok=True)

    for key, value in final_fsa_groups.items():
        value.to_csv(target_dir / f"fsa_round_{int(key)}.csv", index=False)
    gprint("Done.")
