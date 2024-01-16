"""
This module implements functions to ensure reproducibility of results, such as setting the random
seed and loading configuration settings.

It also contains the necessary functions and logic behind parsing the arguments provided to the
pipeline script. For readability, it is isolated from the pipeline script, but it is not used
elsewhere.
"""
import os
import random
import pathlib
import argparse
from typing import Union
import multiprocessing as mp

import torch
import numpy as np

from YACS.yacs import Config


def set_random_seed(seed: int = 0) -> None:
    """
    Set the random seed for all relevant libraries.

    Args:
        seed: The random seed to set.
    """

    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def path_to_project_root() -> pathlib.Path:
    """
    Return the path to the root of the project.

    Returns:
        The path to the root of the project.
    """
    return pathlib.Path(__file__).parent.parent.parent


def load_configuration(
    config_path: Union[str, pathlib.Path] = "default_configuration.yaml"
) -> Config:
    """
    Load the configuration settings from a YAML file.

    Args:
        config_path: The path to the YAML file containing the configuration settings.

    Returns:
        The configuration settings.
    """
    config_path = path_to_project_root() / config_path
    # Config could only be instantiated from dict, yaml filepath, or an argparse.Namespace object
    return Config(str(config_path))


def parse_keyword_arguments() -> argparse.Namespace:
    """
    Parse the keyword arguments passed to the script.

    Returns:
        The keyword arguments passed to the script.
    """
    parser = argparse.ArgumentParser(
        description="Run the pipeline for the project. The pipeline is as follows: "
        "(1) Load the configuration file. "
        "(2) Download the semester data from the Google Drive folder. "
        "(3) Preprocess the data to make it compatible with InferNet. "
        "(4) Aggregate the data into a single file; for example, one file for problem-level data "
        "and one file for each exercises' step-level data. "
        "(5) Train the InferNet model for the problem-level. "
        "(6) Propagate the problem-level rewards to the step-level. "
        "(7) Train the InferNet model for each exercise (step-level) data file that was created in "
        "step (4) above (except for the problem-level data file) using the InferNet model trained "
        "in step (5) above to infer the immediate rewards for each exercise (step-level). "
        "(8) Select the most recent data file (with inferred immediate rewards) "
        "produced as a result "
        "of training InferNet, and store it in the data subdirectory called "
        '"for_policy_induction". '
        "(9) Train the policy induction model using the data files selected in step (8) above.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--step",
        type=int,
        default=1,
        help="The step of the pipeline to run. "
        "1: download the semester data from the Google Drive folder. "
        "2: preprocess the data to make it compatible with InferNet. "
        "3: aggregate the data into a single file; for example, one file for problem-level data "
        "and one file for each exercises' step-level data. "
        "4: train the InferNet model for the problem-level. "
        "5: propagate the problem-level rewards to the step-level. "
        "6: train the InferNet model for each exercise (step-level) data file that was created in "
        "step (4) above (except for the problem-level data file) using the InferNet model trained "
        "in step (5) above to infer the immediate rewards for each exercise (step-level). "
        "7: select the most recent data file (with inferred immediate rewards) "
        "produced as a result "
        "of training InferNet, and store it in the data subdirectory called "
        '"for_policy_induction". '
        "8: train the policy induction model using the data files selected in step (7) above.",
    )
    parser.add_argument(
        "--run_specific",
        default=False,
        action="store_true",
        help="If False, continue with steps after the step that is specified "
        "by the --step argument. "
        "If True, do not continue with steps after the step that is specified "
        "by the --step argument.",
    )
    parser.add_argument(
        "--problem_id",
        type=str,
        default="problem",
        help="The problem ID for which to run the pipeline. "
        "If the step is 1, 2, 3, 4, 5, or 7, this argument is ignored. "
        "If the step is 6, this argument is required. "
        "If the step is 8, this argument is optional. "
        "If the step is 9, this argument is required.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=mp.cpu_count() - 1,
        help="The number of workers (i.e., processes) to use for multiprocessing. "
        "If the step is 1, 2, 3, 4, 5, or 7, this argument is ignored. "
        "If the step is 6, this argument is optional. "
        "If the step is 8, this argument is optional. "
        "If the step is 9, this argument is required.",  # TODO: make this optional?
    )

    parser.add_argument(
        "--config_file_path",
        type=str,
        default="default_configuration.yaml",
        help="The path to the configuration file.",
    )

    arguments = parser.parse_args()
    arguments.num_workers = min(arguments.num_workers, mp.cpu_count() - 1)

    return arguments
