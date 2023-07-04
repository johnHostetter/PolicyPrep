"""
This module implements the pipeline for the project. The pipeline is as follows:
    (1) Download the semester data from the Google Drive folder.
    (2) Preprocess the data to make it compatible with InferNet.
    (3) Aggregate the data into a single file; for example, one file for problem-level data
    and one file for each exercises' step-level data.
    (4) Train the InferNet models.
"""
from src.preprocess.data.download import download_semester_data
from src.preprocess.data.to_infernet_format import iterate_over_semester_data, convert_data_format
from src.preprocess.data.aggregation import aggregate_data_for_inferring_rewards


if __name__ == "__main__":
    # download the data from the Google Drive folder into a subdirectory of the
    # data folder called "raw"
    download_semester_data()

    # preprocess the data to make it compatible with InferNet
    iterate_over_semester_data(
        subdirectory="raw", function_to_perform=convert_data_format
    )

    # aggregate the data into a single file
    aggregate_data_for_inferring_rewards()

    # train the InferNet models
