"""
This module implements functions to ensure reproducibility of results, such as setting the random
seed and loading configuration settings.
"""
import random
import pathlib
from typing import Union

import numpy as np
import tensorflow as tf

from YACS.yacs import Config


def set_random_seed(seed: int = 0):
    """
    Set the random seed for all relevant libraries.

    Args:
        seed: The random seed to set.
    """

    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


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
