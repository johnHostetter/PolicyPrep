"""
This module contains functions for parsing the data into a MDPDataset. The MDPDataset is used to
train the D3RLPy models. This is particularly useful for using "out-of-the-box" offline
reinforcement learning solutions to deploy in the Pyrenees application, such as Double Deep
Q-Networks, Conservative Q-Learning, Soft-Actor-Critic, etc. The MDPDataset is a dataset that
contains the features, actions, rewards, terminals, and next features for each episode. The
MDPDataset is created by (optionally) merging the features dataframe with the decision info
dataframe. The features dataframe contains the features for each step.
"""
from typing import Union

import d3rlpy
import numpy as np
import pandas as pd

from src.utils.reproducibility import load_configuration


def data_frame_to_d3rlpy_dataset(
    features_df: pd.DataFrame,
    problem_id: str,
    decision_info_df: Union[None, pd.DataFrame] = None,
) -> d3rlpy.dataset.MDPDataset:
    """
    Convert the features dataframe to a MDPDataset.

    Args:
        features_df: The features dataframe for the given problem_id.
        problem_id: The problem ID, such as "problem" or "exc137".
        decision_info_df: The decision info dataframe for the given problem_id. Defaults to None. If
            None, then the decision info dataframe is not merged with the features dataframe.

    Returns:
        The MDPDataset for the given problem_id.
    """
    # load the configuration file
    config = load_configuration()
    # make the features dataframe have a consistent column name for the record ID (for merging)
    features_df = features_df.rename(columns={"recordID": "feature_recordID"})
    # if the decision info dataframe is not None, merge it with the features dataframe
    if decision_info_df is not None:
        features_df = features_df.merge(
            decision_info_df, how="inner", on=["feature_recordID", "userID", "problem"]
        )

        if problem_id == "problem":  # problem-level
            column_to_update = "decisionLevel"
        else:  # step-level
            column_to_update = "problem"
        # change the selected column to the problem ID
        features_df = features_df[features_df[column_to_update] == problem_id]
        # first sort the features dataframe by student ID, then by time
        features_df = features_df.sort_values(by=["userID", "time_x"])

    if problem_id == "problem":  # problem-level
        state_features = config.data.features.problem
        # find the terminals (end of episodes)
        terminals = np.zeros(features_df.shape[0])
        students = (
            features_df.userID.unique()
        )  # get the unique IDs for all the students
        transitions_per_user = int(features_df.shape[0] / len(students))
        # replace every nth entry w/ a 1 (end of episode)
        terminals[
            range(transitions_per_user - 1, len(terminals), transitions_per_user)
        ] = 1
    elif problem_id != "problem":  # step-level
        state_features = config.data.features.step
        # find the terminals (end of episodes)
        episode_lengths = features_df.groupby(by=["userID"]).size().values
        terminals = np.zeros(features_df.shape[0])
        idx = episode_lengths[0] - 1
        terminals[idx] = 1
        for episode_length in episode_lengths[1:]:
            idx += episode_length
            terminals[idx] = 1
    else:
        raise ValueError(
            f'problem_id must be either "problem" or the name of an exercise, but got {problem_id}'
        )

    # begin making the MDPDataset
    observations = features_df[state_features].values
    # the action is the decision
    try:
        actions = features_df["decision"].values[:, None]
    except KeyError:  # not 'decision', but 'action'
        actions = features_df["action"].values[:, None]

    # the reward is the inferred reward
    rewards = features_df["inferred_reward"].values[:, None]

    # return the MDPDataset
    return d3rlpy.dataset.MDPDataset(
        observations, actions, rewards, terminals, discrete_action=True
    )
