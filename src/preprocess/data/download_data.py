"""
This module implements functions to download the semester data from the Google Drive folder.
Essentially, this saves us the time from having to collect all relevant training data, as subsequent
steps in the pipeline can simply load the data from the downloaded folder.
"""
import os
import gdown

from src.utils.reproducibility import path_to_project_root, load_configuration

if __name__ == "__main__":
    # load the configuration settings
    config = load_configuration()

    # download the data from the database on Google Drive
    gdown.download_folder(
        url=config.data.folder.name,
        output=str(path_to_project_root()),
    )

    # rename the folder to lowercase
    os.rename(
        src=str(path_to_project_root() / "Data"),
        dst=str(path_to_project_root() / "data"),
    )
