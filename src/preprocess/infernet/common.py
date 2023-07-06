"""
This file contains the common functions used in the InferNet model.
"""
from typing import List, Union

import d3rlpy.dataset
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import TimeDistributed, Dense, Dropout, LeakyReLU

from YACS.yacs import Config
from src.utils.reproducibility import path_to_project_root, load_configuration


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
    if selected_users is None:
        return data[data["userID"] > 161000]  # ignore any user before 161000
    return data[data["userID"].isin(selected_users)]


def model_build(max_ep_length: int, num_sas_features: int) -> Sequential:
    """
    Build the InferNet model.

    Args:
        max_ep_length: The maximum episode length.
        num_sas_features: The number of state and action features.

    Returns:
        The InferNet model.
    """
    model = Sequential()
    model.add(
        TimeDistributed(Dense(256), input_shape=(max_ep_length, num_sas_features))
    )
    model.add(LeakyReLU())
    model.add(Dropout(0.5))
    model.add(TimeDistributed(Dense(256)))
    model.add(LeakyReLU())
    model.add(Dropout(0.5))
    model.add(TimeDistributed(Dense(256)))
    model.add(LeakyReLU())
    model.add(TimeDistributed(Dense(1)))

    def loss_function(true_output, predicted_output):  # TODO: check arg & return types
        """
        InferNet's loss function.

        Args:
            true_output: The true output.
            predicted_output: The predicted output.

        Returns:
            The loss.
        """
        inferred_sum = K.sum(predicted_output, axis=1)
        inferred_sum = tf.reshape(
            inferred_sum, (tf.shape(true_output)[0], tf.shape(true_output)[1])
        )
        return K.mean(K.square(inferred_sum - true_output), axis=-1)

    model.compile(loss=loss_function, optimizer=Adam(learning_rate=0.0001))
    return model


def calc_max_episode_length(
    mdp_dataset: d3rlpy.dataset.MDPDataset
) -> int:
    """
    Calculate the maximum episode length.

    Args:
        mdp_dataset: The Markov Decision Process (MDP) dataset.

    Returns:
        The maximum episode length.
    """
    # -1 because the last step is not a step (it's a terminal state that is not "real")
    return max([len(episode) - 1 for episode in mdp_dataset.episodes])


def normalize_data(
    original_data: pd.DataFrame, file_name: str, columns_to_normalize: List[str]
) -> pd.DataFrame:
    """
    Normalize the data.

    Args:
        original_data: The original data.
        file_name: The file name.
        columns_to_normalize: The columns to normalize (i.e., the features).

    Returns:
        The normalized data.
    """
    # Normalize each column.
    normalized_data = original_data.copy()
    feats, minimums, maximums = [], [], []
    for feature_name in columns_to_normalize:
        max_val = normalized_data[feature_name].max()
        min_val = normalized_data[feature_name].min()
        if min_val == max_val:
            normalized_data[feature_name] = 0.0
        else:
            normalized_data[feature_name] = (original_data[feature_name] - min_val) / (
                max_val - min_val
            )
        feats.append(feature_name)
        minimums.append(min_val)
        maximums.append(max_val)
    normalization_values_df = pd.DataFrame(
        {"feat": feats, "min_val": minimums, "max_val": maximums}
    )
    output_directory = path_to_project_root() / "data" / "normalization_values"
    output_directory.mkdir(parents=True, exist_ok=True)
    normalization_values_df.to_csv(
        output_directory / f"{file_name}.csv", index=False
    )
    return normalized_data


def create_buffer(
    normalized_data: pd.DataFrame,
    user_ids: List[int],
    config: Config,
    is_problem_level: bool = True,
    max_len=None,
) -> List[tuple]:  # TODO: change this buffer to use d3rlpy MDPDataset
    """
    Create the buffer.

    Args:
        normalized_data:
        user_ids:
        config:
        is_problem_level:
        max_len:

    Returns:
        The buffer.
    """
    infer_buffer = []
    for user in user_ids:
        if user in config.training.skip.users:
            continue
        data_st = normalized_data[normalized_data["userID"] == user]
        if is_problem_level:
            feats = data_st.iloc[:-1][list(config.data.features.problem)]
        else:
            feats = data_st.iloc[:-1][list(config.data.features.step)]
        non_feats = data_st.iloc[:-1][list(config.data.features.basic)]

        if is_problem_level and len(feats) != len(config.training.problems):
            # raise ValueError("Problem level episodes should be 12 steps long.")
            print(f"Problem level episodes should be {config.training.problems} steps long.")
            continue  # TODO: figure out why some have less than or more than 12 steps

        actions = data_st["action"].tolist()[:-1]
        feats["action_ps"] = np.array([1.0 if x == "problem" else 0.0 for x in actions])
        feats["action_we"] = np.array([1.0 if x == "example" else 0.0 for x in actions])
        if is_problem_level:  # faded worked example is only for the problem-level
            feats["action_fwe"] = np.array(
                [1.0 if x not in ("problem", "example") else 0.0 for x in actions]
            )

        rewards = data_st["reward"].tolist()
        if np.isnan(rewards[-1]):
            continue  # TODO: figure out why some are bad users w/ delayed reward of NaN
        imm_rews = rewards[:-1]

        if not is_problem_level:
            if len(data_st) - 1 < max_len:
                num_rows, num_cols = feats.shape
                zeros = np.zeros((max_len - num_rows, num_cols))
                data_frame = pd.DataFrame(zeros, columns=feats.columns)
                imm_rews.extend([0.0 for _ in range(max_len - num_rows)])
                feats = pd.concat([feats, data_frame], ignore_index=True)

        feats_np = feats.values
        infer_buffer.append(
            (feats_np, non_feats, imm_rews, rewards[-1], len(rewards) - 1)
        )
    return infer_buffer


def infer_and_save_rewards(
    file_name: str,
    iteration: int,
    infer_buffer: List[tuple],
    max_len: int,
    model: Sequential,
    state_feature_columns: List[str],
    num_state_and_actions: int,
    is_problem_level: bool = True,
) -> None:
    """
    Iterate through the buffer and infer rewards. Then save the rewards.

    Args:
        file_name: The file name.
        iteration: The training iteration.
        infer_buffer: The buffer.
        max_len: The maximum episode length.
        model: The InferNet model.
        state_feature_columns: The state feature columns.
        num_state_and_actions: The number of state and actions.
        is_problem_level: Whether the episode is problem-level.

    Returns:
        None
    """
    result = []
    config = load_configuration()
    for state_transactions in range(len(infer_buffer)):
        states_actions, non_feats, imm_rews, imm_rew_sum, length = infer_buffer[
            state_transactions
        ]
        non_feats = np.array(non_feats)
        states_actions = np.reshape(states_actions, (1, max_len, num_state_and_actions))
        inf_rews = model.predict(states_actions)

        states_actions = np.reshape(states_actions, (max_len, num_state_and_actions))
        if not is_problem_level:  # step-level only operations
            states_actions = states_actions[:length]
            inf_rews = np.reshape(inf_rews, (max_len, 1))
            inf_rews = inf_rews[:length]
        inf_rews = np.reshape(inf_rews, (length, 1))
        all_feats = np.concatenate((non_feats, states_actions, inf_rews), axis=-1)
        for row in all_feats:
            result.append(row)
    result = np.array(result)
    if is_problem_level:
        actions = ["action_ps", "action_we", "action_fwe"]
    else:
        actions = ["action_ps", "action_we"]
    new_columns = actions + ["inferred_reward"]
    result_df = pd.DataFrame(
        result,
        columns=list(config.data.features.basic)
        + list(state_feature_columns)
        + new_columns,
    )
    output_directory = path_to_project_root() / "data" / "with_inferred_rewards"
    output_directory.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(output_directory / f"{file_name}_{iteration}.csv", index=False)
