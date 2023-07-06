"""
This file is used to train the InferNet model for the problem level data.
"""
import os
import random
import time

import numpy as np
import pandas as pd
import tensorflow as tf

from src.preprocess.data.parser import data_frame_to_d3rlpy_dataset

from src.preprocess.infernet.common import (
    read_data,
    model_build,
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
    # configure tensorflow to use the CPU
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    # load the configuration file
    config = load_configuration()
    # set the random seed
    set_random_seed(seed=config.training.seed)
    # determine if the data is problem-level or step-level
    is_problem_level = "problem" in problem_id

    tf.keras.backend.set_floatx("float64")
    original_data = read_data(problem_id, "for_inferring_rewards", selected_users=None)
    mdp_dataset = data_frame_to_d3rlpy_dataset(original_data, problem_id)
    user_ids = original_data["userID"].unique()
    max_len = calc_max_episode_length(mdp_dataset)

    # select the features and actions depending on if the data is problem-level or step-level
    if is_problem_level:
        state_features = config.data.features.problem
        state_actions = config.training.actions.problem
    else:
        state_features = config.data.features.step
        state_actions = config.training.actions.step

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
        max_len=max_len  # max_len is the max episode length; not required for problem-level data
    )

    num_state_and_actions = len(state_features) + len(state_actions)
    print(problem_id)
    print(f"Max episode length is {max_len}")

    # Train Infer Net.
    model = model_build(max_len, len(state_features) + len(state_actions))

    # Train infer_net.
    train_steps = 10001
    print("#####################")
    start_time = time.time()
    losses = []
    for iteration in range(train_steps):
        batch = random.sample(infer_buffer, config.training.data.batch_size)
        states_actions, non_feats, imm_rews, imm_rew_sum, length = list(zip(*batch))
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
        if iteration == 0:
            print(problem_id)
        if iteration % 1000 == 0:
            print(f"Step {iteration}/{train_steps}, loss {loss}")
            print("Training time is", time.time() - start_time, "seconds")
            start_time = time.time()

        if iteration in (1000, 10000):
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
            path_to_models = path_to_project_root() / "models"
            path_to_models.mkdir(parents=True, exist_ok=True)
            model.save(path_to_models / f"{problem_id}_{iteration}.h5")

    print(f"Done training InferNet for {problem_id}.")


if __name__ == "__main__":
    train_infer_net(problem_id="problem")
