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
    if selected_users is None:
        return data[data["userID"] > 161000]  # ignore any user before 161000
    return data[data["userID"].isin(selected_users)]


def build_model(num_sas_features: int) -> (TimeDistributed, torch.optim.Adam):
    """
    Build the InferNet model and its optimizer.

    Args:
        num_sas_features: The number of state and action features.

    Returns:
        The InferNet model and the optimizer.
    """
    neural_network: torch.nn.Sequential = torch.nn.Sequential(
        # torch.nn.LSTM(
        #     input_size=num_sas_features,
        #     hidden_size=20,
        #     num_layers=1,
        #     batch_first=True,
        # ),
        # torch.nn.Linear(in_features=20, out_features=256, bias=True),
        torch.nn.Linear(in_features=num_sas_features, out_features=256, bias=True),
        # TimeDistributed(
        #     module=torch.nn.Linear(in_features=num_sas_features, out_features=256, bias=True),
        #     batch_first=True
        # ),
        torch.nn.PReLU(),
        # torch.nn.LazyBatchNorm1d(),  # lazy version infers the number of input features
        # torch.nn.BatchNorm1d(num_features=128),
        # torch.nn.Dropout(p=0.2),
        torch.nn.Linear(in_features=256, out_features=512, bias=True),
        torch.nn.PReLU(),
        # torch.nn.LazyBatchNorm1d(),  # lazy version infers the number of input features
        # torch.nn.BatchNorm1d(num_features=128),
        # torch.nn.Dropout(p=0.5),
        # torch.nn.Linear(in_features=512, out_features=512, bias=True),
        # torch.nn.ReLU6(),
        # torch.nn.LazyBatchNorm1d(),  # lazy version infers the number of input features
        # torch.nn.BatchNorm1d(num_features=128),
        torch.nn.Dropout(p=0.5),
        torch.nn.Linear(in_features=512, out_features=256, bias=True),
        # TimeDistributed(
        #     module=torch.nn.Linear(in_features=256, out_features=256, bias=True),
        #     batch_first=True
        # ),
        # torch.nn.LeakyReLU(),
        # # torch.nn.LazyBatchNorm1d(),  # lazy version infers the number of input features
        # # torch.nn.BatchNorm1d(num_features=128),
        # torch.nn.Dropout(p=0.5),
        # torch.nn.Linear(in_features=256, out_features=256, bias=True),
        # TimeDistributed(
        #     module=torch.nn.Linear(in_features=256, out_features=128, bias=True),
        #     batch_first=True
        # ),
        torch.nn.PReLU(),
        # torch.nn.Dropout(p=0.5),
        torch.nn.Linear(in_features=256, out_features=3, bias=True),
        # TimeDistributed(
        #     module=torch.nn.Linear(in_features=128, out_features=3, bias=True),
        #     batch_first=True
        # ),
    )

    model = TimeDistributed(module=neural_network, batch_first=True)
    # model = neural_network
    optimizer = torch.optim.Adam(
        model.parameters(), lr=1e-4
    )  # the 'lr' is learning rate
    return model, optimizer


def load_infernet_model(
    path_to_model: Path,
) -> TimeDistributed:  # TODO: rewrite this for PyTorch
    """
    Load the InferNet model.

    Args:
        path_to_model: The path to the model.

    Returns:
        The InferNet model.
    """
    return torch.load(path_to_model)


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
            # normalized_data[feature_name] = (original_data[feature_name] - normalized_data[feature_name].mean()) / normalized_data[feature_name].std()
        feats.append(feature_name)
        minimums.append(min_val)
        maximums.append(max_val)
    normalization_values_df = pd.DataFrame(
        {"feat": feats, "min_val": minimums, "max_val": maximums}
    )
    output_directory = path_to_project_root() / "data" / "normalization_values"
    output_directory.mkdir(parents=True, exist_ok=True)
    normalization_values_df.to_csv(output_directory / f"{file_name}.csv", index=False)
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
    if is_problem_level:
        state_features = list(config.data.features.problem)
        possible_actions = list(config.training.actions.problem)
    else:
        state_features = list(config.data.features.step)
        possible_actions = list(config.training.actions.step)

    # attempting to scale target feature to improve stability
    # normalized_data["reward"] = ((normalized_data["reward"].values - np.nanmin(normalized_data["reward"].values)) / (np.nanmax(normalized_data["reward"].values) - np.nanmin(normalized_data["reward"].values)))
    # normalized_data["reward"] = (normalized_data["reward"].values - np.nanmean(normalized_data["reward"].values)) / np.nanstd(normalized_data["reward"].values)

    # # one-hot encoding of actions; specify their order & add prefix; convert booleans to float
    # actions_df: pd.DataFrame = pd.get_dummies(
    #     normalized_data["action"]
    # )[possible_actions].add_prefix("action_").astype(float)
    # encoded_action_columns: list = list(actions_df.columns)
    # normalized_data: pd.DataFrame = normalized_data.join(actions_df)
    infer_buffer: list = []
    for user in user_ids:
        if user in config.training.skip.users:
            continue
        data_st = normalized_data[normalized_data["userID"] == user]
        if is_problem_level:
            feats = data_st.iloc[:-1][state_features]  # + encoded_action_columns
        else:
            feats = data_st.iloc[:-1][state_features]  # + encoded_action_columns
        non_feats = data_st.iloc[:-1][list(config.data.features.basic)]

        if is_problem_level and len(feats) != len(config.training.problems):
            # raise ValueError("Problem level episodes should be 12 steps long.")
            print(
                f"Issue with user {user}: Problem level episodes "
                f"should be {len(config.training.problems)} steps long."
                f"Skipping this user (and adding them to list of skipped users)."
            )
            with config.unfreeze():
                config.training.skip.users = tuple(
                    list(config.training.skip.users) + [user]
                )
            continue  # TODO: figure out why some have less than or more than 12 steps

        actions = data_st["action"].tolist()[:-1]
        new_columns = {}
        for possible_action in possible_actions:
            new_columns[f"action_{possible_action}"] = np.array(
                [1.0 if x == possible_action else 0.0 for x in actions]
            )
        # new_columns = {
        #     "action_ps": np.array([1.0 if "problem" in x else 0.0 for x in actions]),
        #     "action_we": np.array([1.0 if "example" in x else 0.0 for x in actions]),
        # }
        # if is_problem_level:  # faded worked example is only for the problem-level
        #     new_columns["action_fwe"] = np.array(
        #         [1.0 if ("problem" not in x and "example" not in x) else 0.0 for x in actions]
        #     )
        feats = pd.concat(
            [
                feats,
                pd.DataFrame(
                    new_columns, index=feats.index, columns=list(new_columns.keys())
                ),
            ],
            axis=1,
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
    problem_id: str,
    iteration: int,
    infer_buffer: List[tuple],
    max_len: int,
    # model: Sequential,
    model: TimeDistributed,
    state_feature_columns: List[str],
    num_state_and_actions: int,
    is_problem_level: bool = True,
    inferred_reward_column_name: str = "inferred_reward",
    save_inferred_rewards: bool = True,
) -> pd.DataFrame:
    """
    Iterate through the buffer and infer rewards. Then save the rewards.

    Args:
        problem_id: The problem ID, or "problem" if the data is problem-level.
        iteration: The training iteration.
        infer_buffer: The buffer.
        max_len: The maximum episode length.
        model: The InferNet model.
        state_feature_columns: The state feature columns.
        num_state_and_actions: The number of state and actions.
        is_problem_level: Whether the episode is problem-level.
        inferred_reward_column_name: The name of the inferred reward column. Defaults to
        "inferred_reward". Useful for when hypothetical actions are used.
    Returns:
        None
    """
    # result = []
    config = load_configuration()
    # Preallocate the result list
    result = []

    # Prepare inputs for batched prediction
    batch_states_actions = []
    batch_non_feats = []
    batch_lengths = []

    for states_actions, non_feats, _, _, length in infer_buffer:
        non_feats = np.array(non_feats)
        batch_states_actions.append(
            np.reshape(states_actions, (max_len, num_state_and_actions))
        )
        batch_non_feats.append(non_feats)
        batch_lengths.append(length)

    batch_states_actions = np.array(batch_states_actions)
    batch_non_feats = np.array(batch_non_feats)
    batch_lengths = np.array(batch_lengths)

    # Perform batched predictions
    # inf_rews = model.predict(batch_states_actions, batch_size=32, verbose=0)
    inf_rews = (
        model.cuda()(torch.Tensor(batch_states_actions[:, :, :-3]).cuda())
        .cpu()
        .detach()
        .numpy()
    )
    # inf_rews = tf.squeeze(inf_rews, axis=-1)

    for i in range(len(infer_buffer)):
        states_actions = batch_states_actions[i]
        non_feats = batch_non_feats[i]
        length = batch_lengths[i]
        inf_rews_i = inf_rews[i]

        if not is_problem_level:  # step-level only operations
            states_actions = states_actions[:length]
            inf_rews_i = inf_rews_i[:length]

        all_feats = np.concatenate(
            (non_feats[:length], states_actions, inf_rews_i), axis=-1
        )
        result.extend(all_feats)

    # Convert the result list to a NumPy array
    result = np.array(result)

    # for state_transactions in range(len(infer_buffer)):
    #     states_actions, non_feats, _, _, length = infer_buffer[state_transactions]
    #     non_feats = np.array(non_feats)
    #     states_actions = np.reshape(states_actions, (1, max_len, num_state_and_actions))
    #     inf_rews = model.predict(states_actions, verbose=0)
    #
    #     states_actions = np.reshape(states_actions, (max_len, num_state_and_actions))
    #     if not is_problem_level:  # step-level only operations
    #         states_actions = states_actions[:length]
    #         inf_rews = np.reshape(inf_rews, (max_len, 1))
    #         inf_rews = inf_rews[:length]
    #     inf_rews = np.reshape(inf_rews, (length, 1))
    #     all_feats = np.concatenate((non_feats, states_actions, inf_rews), axis=-1)
    #     for row in all_feats:
    #         result.append(row)
    # result = np.array(result)
    if is_problem_level:
        actions = ["action_ps", "action_we", "action_fwe"]
    else:
        actions = ["action_ps", "action_we"]
    new_columns = actions + [inferred_reward_column_name]
    result_df = pd.DataFrame(
        result,
        columns=list(config.data.features.basic)
        + list(state_feature_columns)
        + new_columns,
    )
    if save_inferred_rewards:
        output_directory = path_to_project_root() / "data" / "with_inferred_rewards"
        output_directory.mkdir(parents=True, exist_ok=True)
        result_df.to_csv(
            output_directory / f"{problem_id}_{iteration}.csv", index=False
        )

    return result_df
