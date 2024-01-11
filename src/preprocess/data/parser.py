"""
This module contains functions for parsing the data into a MDPDataset. The MDPDataset is used to
train the D3RLPy models. This is particularly useful for using "out-of-the-box" offline
reinforcement learning solutions to deploy in the Pyrenees application, such as Double Deep
Q-Networks, Conservative Q-Learning, Soft-Actor-Critic, etc. The MDPDataset is a dataset that
contains the features, actions, rewards, terminals, and next features for each episode. The
MDPDataset is created by (optionally) merging the features dataframe with the decision info
dataframe. The features dataframe contains the features for each step.
"""
import warnings
from typing import Union, List

import d3rlpy
import numpy as np
import pandas as pd
from alive_progress import alive_it
from colorama import (
    init as colorama_init,
)  # for cross-platform colored text in the terminal
from colorama import Fore, Style  # for cross-platform colored text in the terminal

from YACS.yacs import Config
from src.utilities.reproducibility import load_configuration

colorama_init()  # initialize colorama


def data_frame_to_d3rlpy_dataset(
    features_df: pd.DataFrame,
    problem_id: str,
    padding: bool = False,
    decision_info_df: Union[None, pd.DataFrame] = None,
    config: Union[None, Config] = None,
) -> (d3rlpy.dataset.MDPDataset, pd.DataFrame):
    """
    Convert the features dataframe to a MDPDataset.

    Args:
        features_df: The features dataframe for the given problem_id.
        problem_id: The problem ID, such as "problem" or "exc137".
        padding: Whether to pad episodes with zeros. Defaults to False; InferNet expects padding.
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

    # find if anyone has nan for delayed reward
    # (has no effect if run *after* rewards have been inferred, as intended)
    group_by = list(features_df.groupby(by="userID"))
    delayed_rewards = [
        group_by[idx][1].reward.iloc[-1]
        for idx in range(len(features_df.userID.unique()))
    ]
    num_users_with_nan = np.sum(np.isnan(delayed_rewards))
    if num_users_with_nan > 0:
        warnings.warn(
            f"Number of users with NaN delayed rewards: {np.sum(np.isnan(delayed_rewards))}"
        )

    # find the users with NaN delayed rewards
    users_with_nan = [
        group_by[idx][0]
        for idx in range(len(features_df.userID.unique()))
        if np.isnan(delayed_rewards[idx])
    ]
    # check we found them all
    assert len(users_with_nan) == num_users_with_nan, (
        f"Number of users with NaN delayed rewards: {num_users_with_nan} "
        f"!= number of users with NaN delayed rewards: {len(users_with_nan)}"
    )
    # remove them from the dataset
    features_df = features_df[~features_df["userID"].isin(users_with_nan)]
    # check that they are removed
    group_by = list(features_df.groupby(by="userID"))
    assert (
        np.sum(
            np.isnan(
                [
                    group_by[idx][1].reward.iloc[-1]
                    for idx in range(len(features_df.userID.unique()))
                ]
            )
        )
        == 0
    ), "There are still users with NaN delayed rewards"

    assert (
        len(features_df)
        > 0  # check records remain after eliminating users with NaN rewards
    ), f"({problem_id}) The features_df has length {len(features_df)}; it should be greater than 0"

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
    grouped_features_by_user: List[pd.DataFrame] = [
        group_df for userID, group_df in features_df.groupby(by="userID")
    ]

    def calculate_all_episode_lengths(
        grouped_features_by_user: List[pd.DataFrame],
    ) -> List[int]:
        """
        Calculate the length of each episode for each user.

        Args:
            grouped_features_by_user: The features dataframe grouped by user ID.

        Returns:
            The length of each episode for each user.
        """
        return [len(group_df) for group_df in grouped_features_by_user]

    episode_lengths = calculate_all_episode_lengths(grouped_features_by_user)
    if min(episode_lengths) != max(episode_lengths) and padding:
        print(
            f"{Fore.YELLOW}"
            f"Warning: The episode lengths are not all the same. This is expected for step-level "
            f"policies. \nThe min episode length is {min(episode_lengths)} and the max episode "
            f"length is {max(episode_lengths)} for {problem_id}. Padding the episodes with zeros..."
            f"{Style.RESET_ALL}"
        )

        # then we must pad the episodes with zeros, we will modify the features_df in place
        max_episode_length = len(max(grouped_features_by_user, key=len))

        # create a new dataframe
        padded_user_features_dfs: List[pd.DataFrame] = []

        # iterate over the users and populate the new dataframe
        for user_df in alive_it(grouped_features_by_user):
            # get the user ID for later
            user_ids: np.ndarray = user_df["userID"].unique()
            assert (
                len(user_ids) == 1
            ), f"Expected the user dataframe to have only one user ID, but got {user_ids}"
            user_id: int = int(user_ids[0])

            # find the number of rows to add
            num_rows_to_add = max_episode_length - len(user_df)
            # create a dataframe with the correct number of rows
            zeros_df = pd.DataFrame(
                np.zeros((num_rows_to_add, len(user_df.columns))),
                columns=user_df.columns,
            )
            # append the zeros dataframe to the user dataframe
            user_df = pd.concat(
                # [user_df, zeros_df.replace(0, np.nan, inplace=False)],
                [user_df, zeros_df],
                ignore_index=True,
                sort=False,
            )
            # update the user ID column
            user_df["userID"] = user_id
            # append the user dataframe to the list of padded users' features dataframes
            padded_user_features_dfs.append(user_df)
            # padded_features_df = pd.concat(
            #     [padded_features_df, user_df], ignore_index=True, sort=False, axis=0
            # )

        # concatenate the padded user features dataframes
        padded_features_df = pd.concat(
            padded_user_features_dfs, ignore_index=True, sort=False, axis=0
        )

        # check that the operation was successful
        assert min(
            [len(group) for _, group in padded_features_df.groupby(by="userID")]
        ) == max([len(group) for _, group in padded_features_df.groupby(by="userID")])
        assert (
            len(padded_features_df)
            == len(features_df.userID.unique()) * max_episode_length
        ), (
            f"Expected the padded features dataframe to have length "
            f"{len(features_df) * len(grouped_features_by_user)}, "
            f"but got {len(padded_features_df)}"
        )

        # replace the features dataframe with the padded features dataframe
        features_df = padded_features_df

        # update the episode lengths to reflect the padding
        episode_lengths = calculate_all_episode_lengths(
            [group_df for userID, group_df in features_df.groupby(by="userID")]
        )

        print(
            f"{Fore.GREEN}"
            f"Finished padding the episodes with zeros for {problem_id}. "
            f"{Style.RESET_ALL}"
        )

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
    # 0 is problem (or PSFWE), 1 is example (WEFWE), 2 is step_decision, 3 is no-action
    # PSFWE -> problem, WEFWE -> example
    # these values were created to show the exercise as a problem-solving (or worked example),
    # but record the data as if it were a faded worked example (step_decision); this was done
    # because inverse RL was not working with the data as a problem-solving (or worked example)
    features_df["action"] = features_df["action"].replace(
        to_replace=[
            "problem",
            "PSFWE",
            "example",
            "WEFWE",
            "step_decision",
            "no-action",
        ],
        value=[0, 0, 1, 1, 2, 3],
    )
    actions = features_df[action_column].values[:, None]

    # the reward is the inferred reward
    try:
        rewards = features_df["inferred_reward"].values[:, None]
    except KeyError:  # not 'inferred_reward', but 'reward'
        # TODO: replace np.nan with the actual reward; for now, just use 0.0
        rewards = features_df["reward"].replace(np.nan, 0.0).values[:, None]

    return (
        d3rlpy.dataset.MDPDataset(
            observations, actions, rewards, terminals, discrete_action=True
        ),
        features_df,
    )
