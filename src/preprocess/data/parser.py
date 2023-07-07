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

from YACS.yacs import Config
from src.utils.reproducibility import load_configuration


def data_frame_to_d3rlpy_dataset(
    features_df: pd.DataFrame,
    problem_id: str,
    decision_info_df: Union[None, pd.DataFrame] = None,
    config: Union[None, Config] = None,
) -> d3rlpy.dataset.MDPDataset:
    """
    Convert the features dataframe to a MDPDataset.

    Args:
        features_df: The features dataframe for the given problem_id.
        problem_id: The problem ID, such as "problem" or "exc137".
        decision_info_df: The decision info dataframe for the given problem_id. Defaults to None. If
        None, then the decision info dataframe is not merged with the features dataframe.
        config: The configuration file. Defaults to None. If None, then the default configuration
        file is loaded.

    Returns:
        The MDPDataset for the given problem_id.
    """
    if config is None:
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

    # drop the users that we should skip (e.g., the users that have invalid data)
    features_df = features_df[~features_df["userID"].isin(config.training.skip.users)]

    if problem_id == "problem":  # problem-level
        state_features = config.data.features.problem
    elif problem_id != "problem":  # step-level
        state_features = config.data.features.step
    else:
        raise ValueError(
            f'problem_id must be either "problem" or the name of an exercise, but got {problem_id}'
        )

    # --- begin making the MDPDataset ---
    # find the terminals (end of episodes)
    episode_lengths = features_df.groupby(by=["userID"]).size().values
    terminals = np.zeros(features_df.shape[0])
    idx = episode_lengths[0] - 1
    terminals[idx] = 1
    for episode_length in episode_lengths[1:]:
        idx += episode_length
        terminals[idx] = 1
    # drop all the columns that are not state features
    observations = features_df[list(state_features)].values
    if "decision" in features_df.columns:
        action_column = "decision"  # the action is the decision
    elif "action" in features_df.columns:
        action_column = "action"
    else:  # neither "decision" nor "action" is in the columns
        raise ValueError(
            f'features_df must have either "decision" or action_column as a column, '
            f"but got {features_df.columns}"
        )
    # replace all the NaNs with "no-action"
    features_df[action_column] = features_df[action_column].replace(np.nan, "no-action")
    # 0 is problem (or PSFWE), 1 is step_decision, 2 is example (WEFWE), 3 is no-action
    # PSFWE -> problem, WEFWE -> example
    # these values were created to show the exercise as a problem-solving (or worked example),
    # but record the data as if it were a faded worked example (step_decision); this was done
    # because inverse RL was not working with the data as a problem-solving (or worked example)
    features_df["action"] = features_df["action"].replace(
        to_replace=[
            "problem",
            "step_decision",
            "example",
            "PSFWE",
            "WEFWE",
            "no-action",
        ],
        value=[0, 1, 2, 0, 2, 3],
    )
    actions = features_df[action_column].values[:, None]

    # the reward is the inferred reward
    try:
        rewards = features_df["inferred_reward"].values[:, None]
    except KeyError:  # not 'inferred_reward', but 'reward'
        # TODO: replace np.nan with the actual reward; for now, just use 0.0
        rewards = features_df["reward"].replace(np.nan, 0.0).values[:, None]

    return d3rlpy.dataset.MDPDataset(
        observations, actions, rewards, terminals, discrete_action=True
    )
