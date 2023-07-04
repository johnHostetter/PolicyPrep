"""
This file contains the common functions used in the InferNet model.
"""
from typing import List

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import TimeDistributed, Dense, Dropout, LeakyReLU

from YACS.yacs import Config
from src.utils.reproducibility import path_to_project_root


def read_data(file_name: str) -> pd.DataFrame:
    """
    Read the data from the csv file.

    Args:
        file_name: The name of the file to read.

    Returns:
        The data from the csv file.
    """
    data_path = (
        path_to_project_root() / "data" / "for_inferring_rewards" / f"{file_name}.csv"
    )
    data = pd.read_csv(data_path, header=0)
    return data[data["userID"] > 161000]  # ignore any user before 161000


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
    original_data: pd.DataFrame, user_ids: List[int], config: Config
) -> int:
    """
    Calculate the maximum episode length.

    Args:
        original_data: The original data.
        user_ids: A list of user ids.
        config: The configuration file.

    Returns:
        The maximum episode length.
    """
    max_len = 0
    for user in user_ids:
        if user in config.training.skip.users:
            continue
        data_st = original_data[original_data["userID"] == user]
        if len(data_st) > max_len:
            max_len = len(data_st)
    max_len -= 1  # the last row is not *really* a step
    return max_len


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
    normalization_values_df.to_csv(f"normalization_values/{file_name}.csv", index=False)
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

        if is_problem_level and len(feats) != 12:
            # raise ValueError("Problem level episodes should be 12 steps long.")
            print("Problem level episodes should be 12 steps long.")
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
                feats = feats.append(data_frame, ignore_index=True)

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
    normalized_data: pd.DataFrame,
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
        normalized_data: The normalized data.
        num_state_and_actions: The number of state and actions.
        is_problem_level: Whether the episode is problem-level.

    Returns:
        None
    """
    result = []
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
        columns=normalized_data.columns.tolist() + new_columns,
    )
    output_directory = path_to_project_root() / "with_inferred_rewards"
    output_directory.mkdir(parents=True, exist_ok=True)

    result_df.to_csv(output_directory / f"{file_name}_{iteration}.csv", index=False)
