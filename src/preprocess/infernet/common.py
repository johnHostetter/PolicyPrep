"""
This file contains the common functions used in the InferNet model.
"""
from pathlib import Path
from typing import List, Union

import torch
import numpy as np
import pandas as pd
import d3rlpy.dataset

from YACS.yacs import Config
from src.utilities.wrappers import TimeDistributed
from src.utilities.reproducibility import path_to_project_root, load_configuration


def read_data(
    file_name: str, subdirectory, selected_users: Union[None, List[str]]
) -> pd.DataFrame:
    """
    Read the data from the csv file.

    Args:
        file_name: The name of the file to read.
        subdirectory: The subdirectory where the file is located, a child of the data directory.
        selected_users: The list of users to select. If None, then all users are selected that
        occur after the first 161000 users. In other words, we ignore all students that were
        using the tutor before Spring 2016.

    Returns:
        The data from the csv file.
    """
    if ".csv" not in file_name:
        file_name += ".csv"
    data_path = path_to_project_root() / "data" / subdirectory / file_name
    data = pd.read_csv(data_path, header=0)
    # if selected_users is None:
    #     return data[data["userID"] > 161000]  # ignore any user before 161000
    # return data[data["userID"].isin(selected_users)]
    return data


def build_model(in_features: int, hidden_dim=128) -> TimeDistributed:
    """
    Build a time-distributed neural network.

    Args:
        in_features: The number of input features.
        hidden_dim: The number of hidden dimensions.

    Returns:
        A time-distributed neural network.
    """
    neural_network: torch.nn.Sequential = torch.nn.Sequential(
        torch.nn.Linear(in_features=in_features, out_features=hidden_dim, bias=True),
        torch.nn.PReLU(),
        torch.nn.Linear(in_features=hidden_dim, out_features=hidden_dim, bias=True),
        torch.nn.PReLU(),
        torch.nn.PReLU(),
        torch.nn.Linear(in_features=hidden_dim, out_features=1, bias=True),
    )
    return TimeDistributed(module=neural_network, batch_first=True)


def calc_max_episode_length(mdp_dataset: d3rlpy.dataset.MDPDataset) -> int:
    """
    Calculate the maximum episode length.

    Args:
        mdp_dataset: The Markov Decision Process (MDP) dataset.

    Returns:
        The maximum episode length.
    """
    # -1 because the last step is not a step (it's a terminal state that is not "real")
    return max(len(episode) - 1 for episode in mdp_dataset.episodes)
