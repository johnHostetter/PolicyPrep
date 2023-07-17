"""
This file is used to train the InferNet model for the problem level data.
"""
import os
import time
import random
import argparse
import multiprocessing as mp
from typing import Tuple, List

import numpy as np
import pandas as pd
from numba import cuda
import tensorflow as tf

from YACS.yacs import Config
from src.preprocess.data.parser import data_frame_to_d3rlpy_dataset

from src.preprocess.infernet.common import (
    read_data,
    build_model,
    calc_max_episode_length,
    normalize_data,
    create_buffer,
    infer_and_save_rewards,
)
from src.utils.reproducibility import (
    load_configuration,
    set_random_seed,
    path_to_project_root,
)


def train_infer_net(problem_id: str) -> None:
    """
    Train the InferNet model for the problem level data if "problem" is in problem_id.
    Otherwise, train the InferNet model for the step level data.

    Args:
        problem_id: A string that is either "problem" or the name of an exercise.

    Returns:
        None
    """
    # see if a GPU is available
    physical_devices = tf.config.list_physical_devices("GPU")
    if len(physical_devices) > 0:
        # configure tensorflow to use the GPU
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"

        # set the memory growth to true
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

        # print the device name
        print(f"Running on {physical_devices[0]}")
    else:
        # configure tensorflow to use the CPU
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    # disable the TensorFlow output messages
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    # load the configuration file
    config = load_configuration()
    # set the random seed
    set_random_seed(seed=config.training.seed)
    is_problem_level, max_len, original_data, user_ids = infernet_setup(problem_id)

    # select the features and actions depending on if the data is problem-level or step-level
    state_features, possible_actions = get_features_and_actions(
        config, is_problem_level
    )

    # normalize the data
    normalized_data = normalize_data(
        original_data, problem_id, columns_to_normalize=state_features
    )
    # create the buffer to train InferNet from
    infer_buffer = create_buffer(
        normalized_data,
        user_ids,
        config,
        is_problem_level=is_problem_level,
        max_len=max_len,  # max_len is the max episode length; not required for problem-level data
    )

    num_state_and_actions = len(state_features) + len(possible_actions)
    print(f"{problem_id}: Max episode length is {max_len}")

    model = build_model(max_len, len(state_features) + len(possible_actions))

    # Train the InferNet model
    print("#####################")
    start_time = time.time()
    losses = []
    for iteration in range(config.training.data.num_iterations + 1):
        batch = random.sample(infer_buffer, config.training.data.batch_size)
        states_actions, _, _, imm_rew_sum, _ = list(zip(*batch))
        states_actions = np.reshape(
            states_actions,
            (config.training.data.batch_size, max_len, num_state_and_actions),
        )
        imm_rew_sum = np.reshape(imm_rew_sum, (config.training.data.batch_size, 1))

        hist = model.fit(
            states_actions,
            imm_rew_sum,
            epochs=1,
            batch_size=config.training.data.batch_size,
            verbose=0,
        )
        loss = hist.history["loss"][0]
        losses.append(loss)
        if iteration > 0 and iteration % config.training.data.checkpoint == 0:
            print(
                f"Step {iteration}/{config.training.data.num_iterations}, loss {loss}"
            )
            print("Training time is", time.time() - start_time, "seconds")
            start_time = time.time()

            # Infer the rewards for the data and save the data.
            if is_problem_level:
                state_feature_columns = config.data.features.problem
            else:
                state_feature_columns = config.data.features.step

            infer_and_save_rewards(
                problem_id,
                iteration,
                infer_buffer,
                max_len,
                model,
                state_feature_columns,
                num_state_and_actions,
                is_problem_level=is_problem_level,
            )

            loss_df = pd.DataFrame({"loss": losses})
            # save the loss data to generate the loss plot
            path_to_figures = path_to_project_root() / "figures"
            path_to_figures.mkdir(parents=True, exist_ok=True)
            loss_df.to_csv(
                path_to_figures / f"loss_{problem_id}_{iteration}.csv", index=False
            )
            # save the model
            path_to_models = path_to_project_root() / "models" / "infernet"
            path_to_models.mkdir(parents=True, exist_ok=True)
            model.save(path_to_models / f"{problem_id}_{iteration}.h5")

    print(f"Done training InferNet for {problem_id}.")


def infernet_setup(problem_id: str) -> Tuple[bool, int, pd.DataFrame, np.ndarray]:
    """
    Set up the InferNet model for the problem level data if "problem" is in problem_id. Otherwise,
    set up the InferNet model for the step level data.

    Args:
        problem_id:

    Returns:
        A tuple containing the following:
            is_problem_level: A boolean indicating if the data is problem-level or step-level.
            max_len: The maximum episode length.
            original_data: The original data.
            user_ids: The user IDs.
    """
    # determine if the data is problem-level or step-level
    is_problem_level = "problem" in problem_id
    tf.keras.backend.set_floatx("float64")
    original_data = read_data(problem_id, "for_inferring_rewards", selected_users=None)
    mdp_dataset = data_frame_to_d3rlpy_dataset(original_data, problem_id)
    user_ids = original_data["userID"].unique()
    max_len = calc_max_episode_length(mdp_dataset)
    return is_problem_level, max_len, original_data, user_ids


def get_features_and_actions(
    config: Config, is_problem_level: bool
) -> Tuple[List[str], List[str]]:
    """
    Get the features and actions depending on if the data is problem-level or step-level.

    Args:
        config: The configuration object.
        is_problem_level: A boolean indicating if the data is problem-level or step-level.

    Returns:
        The features and actions.
    """
    if is_problem_level:
        return config.data.features.problem, config.training.actions.problem
    return config.data.features.step, config.training.actions.step


def train_step_level_models(
    args: argparse.Namespace, config: Config, increase_num_workers: bool = False
) -> None:
    """
    Train the InferNet models for all step-level problems.

    Args:
        args: The command line arguments.
        config: The configuration object.
        increase_num_workers: A boolean indicating if the number of workers should be increased.
        This can be enabled to speed up training, but it may cause memory issues. Defaults to False.

    Returns:
        None
    """
    physical_devices = tf.config.list_physical_devices("GPU")
    if increase_num_workers and len(physical_devices) > 0:
        # temporarily limit the number of workers to the number of GPUs available
        device = cuda.get_current_device()
        num_workers = getattr(device, "MULTIPROCESSOR_COUNT", 1)
    else:
        num_workers = args.num_workers
    with mp.Pool(processes=num_workers) as pool:
        for problem_id in config.training.problems:
            print(f"Training the InferNet model for {problem_id}...")
            if problem_id not in config.training.skip.problems:
                pool.apply_async(train_infer_net, args=(f"{problem_id}(w)",))
        pool.close()
        pool.join()
    print("All processes finished for training step-level InferNet models.")


if __name__ == "__main__":
    train_infer_net(problem_id="problem")
