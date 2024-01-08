"""
This module implements functions to download the semester data from the Google Drive folder.
Essentially, this saves us the time from having to collect all relevant training data, as subsequent
steps in the pipeline can simply load the data from the downloaded folder.
"""
import gdown

from src.utilities.reproducibility import path_to_project_root, load_configuration


def download_semester_data():
    """
    Download the semester data from the Google Drive folder.
    """
    # load the configuration settings
    config = load_configuration()

    # make the folder to store the data in
    (path_to_project_root() / "data" / "raw").mkdir(parents=True, exist_ok=True)

    # download the data from the database on Google Drive
    gdown.download_folder(
        url=config.data.folder.name,
        output=str(path_to_project_root() / "data" / "raw"),
    )


if __name__ == "__main__":
    download_semester_data()
