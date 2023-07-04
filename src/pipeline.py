"""
This module implements the pipeline for the project. The pipeline is as follows:
    (1) Download the semester data from the Google Drive folder.
    (2) Preprocess the data to make it compatible with InferNet.
    (3) Aggregate the data into a single file; for example, one file for problem-level data
    and one file for each exercises' step-level data.
    (4) Train the InferNet models.
"""
from src.utils.reproducibility import load_configuration
from src.preprocess.infernet.train import train_infer_net
from src.preprocess.data.download import download_semester_data
from src.preprocess.data.to_infernet_format import iterate_over_semester_data, convert_data_format
from src.preprocess.data.aggregation import aggregate_data_for_inferring_rewards

if __name__ == "__main__":
    # load the configuration file
    config = load_configuration()

    # download the data from the Google Drive folder into a subdirectory of the
    # data folder called "raw"
    download_semester_data()

    # preprocess the data to make it compatible with InferNet
    iterate_over_semester_data(
        subdirectory="raw", function_to_perform=convert_data_format
    )

    # aggregate the data into a single file
    aggregate_data_for_inferring_rewards()

    # train the InferNet model for the problem level data
    train_infer_net(problem_id="problem")

    # propagate problem-level rewards to step-level rewards
    # propagate_problem_level_rewards_to_step_level()

    # train an InferNet model for each exercise
    for problem_id in config.training.problems:
        if problem_id not in config.training.skip.problems:
            train_infer_net(problem_id=problem_id)
