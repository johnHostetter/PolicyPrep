import warnings
from typing import List
from pathlib import Path

from alive_progress import alive_bar
from colorama import (
    init as colorama_init,
)  # for cross-platform colored text in the terminal
from colorama import Fore, Style  # for cross-platform colored text in the terminal

import pandas as pd

from YACS.yacs import Config
from src.preprocess.data.to_infernet_format import make_year_and_semester_int

colorama_init()  # initialize colorama


def append_year_and_semester_to_user_id(
    semester_folder: Path, semester_name: str, config: Config
) -> None:
    """
    Append the year and semester to the user IDs if they are not already there. This is necessary
    because the user IDs are not unique across semesters, so we need to add the year and semester
    to the user IDs to make them unique.

    Args:
        semester_folder:
        semester_name:
        config:

    Returns:

    """
    # recursively find all the .csv files in the semester folder
    csv_files: List[Path] = list(semester_folder.rglob("*.csv"))
    with alive_bar(len(csv_files), title=f"{semester_name}") as bar:
        for csv_file in csv_files:
            print(f"Processing {csv_file.name}...")
            try:
                df = pd.read_csv(csv_file)
                year_int, semester_int = make_year_and_semester_int(semester_name)

                # prepare the file location for the new file (replace "raw" with "clean")
                # either the data will have its userID column appended with the year and semester
                # or it will be copied to the clean folder as-is
                new_file_location = Path(str(csv_file).replace("raw", "clean"))
                assert (
                    new_file_location.parent.exists()
                )  # sanity check, should always exist (created in previous step)
                new_file_location.parent.mkdir(parents=True, exist_ok=True)

                # if the minimum user ID is less than the minimum user ID for the semester, then add the
                # year_int and semester_int to the user IDs
                if (
                    "userID" in df.columns
                    and df["userID"].min() < year_int + semester_int
                ):
                    df["userID"] = df["userID"] + year_int + semester_int
                    print(
                        f"{Fore.GREEN}{semester_name}: "
                        f"Appended year and semester to user IDs in "
                        f"{new_file_location.name}{Style.RESET_ALL}"
                    )
                else:
                    print(
                        f"{Fore.GREEN}{semester_name}: "
                        f"Copied {new_file_location.name} as-is{Style.RESET_ALL}"
                    )

                # save the new file whether the user IDs were appended with the year and semester
                df.to_csv(new_file_location, index=False)
            except pd.errors.EmptyDataError:
                warnings.warn(f"Skipping {csv_file.name} (empty file)...")
                continue
            bar()
