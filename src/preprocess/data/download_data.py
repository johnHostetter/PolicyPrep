"""
This module implements functions to download the semester data from the Google Drive folder. Essentially, this saves us
the time from having to collect all relevant training data, as subsequent steps in the pipeline can simply load the data
from the downloaded folder.
"""
import gdown

from utils.reproducibility import path_to_project_root

# import pandas as pd

# download the data from the database on Google Drive

gdown.download_folder(
    url="https://drive.google.com/drive/folders/1GhSpf6jIuzsBuCYf6bqheV58WioxyoxI",
    output=str(path_to_project_root())
)
