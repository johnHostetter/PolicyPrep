"""
This file is used to train the InferNet model for the problem level data.
"""
import os
import random
import time

import numpy as np
import pandas as pd
import tensorflow as tf

from src.preprocess.infernet.common import (
    read_data,
    model_build,
    calc_max_episode_length,
    normalize_data,
    create_buffer,
    infer_and_save_rewards,
)
from src.utils.reproducibility import load_configuration, set_random_seed, path_to_project_root


def train_infer_net(problem_id):
    # configure tensorflow to use the CPU
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    # load the configuration file
    config = load_configuration()
    # set the random seed
    set_random_seed(seed=config.training.seed)
    # determine if the data is problem-level or step-level
    is_problem_level = "problem" in problem_id

    tf.keras.backend.set_floatx("float64")
    original_data = read_data(problem_id)
    user_ids = original_data["userID"].unique()
    max_len = calc_max_episode_length(original_data, user_ids, config)

    num_state_features = len(config.data.features.problem)
    num_actions = len(config.training.actions.problem)
    num_state_and_actions = num_state_features + num_actions
    print(problem_id)
    print(f"Max episode length is {max_len}")

    if is_problem_level:
        normalized_data = normalize_data(
            original_data, problem_id, columns_to_normalize=config.data.features.problem
        )
        infer_buffer = create_buffer(
            normalized_data, user_ids, config, is_problem_level=is_problem_level, max_len=None
        )
    else:
        normalized_data = normalize_data(
            original_data, problem_id, columns_to_normalize=config.data.features.step
        )
        infer_buffer = create_buffer(
            normalized_data, user_ids, config, is_problem_level=is_problem_level, max_len=max_len
        )

    # Train Infer Net.
    model = model_build(max_len, num_state_features + num_actions)

    # Train infer_net.
    train_steps = 10001
    print("#####################")
    start_time = time.time()
    losses = []
    for iteration in range(train_steps):
        batch = random.sample(infer_buffer, config.training.data.batch_size)
        states_actions, non_feats, imm_rews, imm_rew_sum, length = list(zip(*batch))
        states_actions = np.reshape(
            states_actions, (config.training.data.batch_size, max_len, num_state_and_actions)
        )
        imm_rew_sum = np.reshape(imm_rew_sum, (config.training.data.batch_size, 1))

        hist = model.fit(
            states_actions, imm_rew_sum, epochs=1,
            batch_size=config.training.data.batch_size, verbose=0
        )
        loss = hist.history["loss"][0]
        losses.append(loss)
        if iteration == 0:
            print(problem_id)
        if iteration % 1000 == 0:
            print(f"Step {iteration}/{train_steps}, loss {loss}")
            print("Training time is", time.time() - start_time, "seconds")
            start_time = time.time()

        if iteration in (5000, 10000):
            # Infer the rewards for the data and save the data.
            infer_and_save_rewards(
                problem_id,
                iteration,
                infer_buffer,
                max_len,
                model,
                normalized_data,
                num_state_and_actions,
                is_problem_level=is_problem_level,
            )

            df = pd.DataFrame({"loss": losses})
            path_to_figures = path_to_project_root() / "figures"
            path_to_models = path_to_project_root() / "models"
            path_to_figures.mkdir(parents=True, exist_ok=True)
            path_to_models.mkdir(parents=True, exist_ok=True)
            df.to_csv(f"figures/loss_{problem_id}_{iteration}.csv", index=False)
            model.save(f"models/model_{problem_id}_{iteration}.h5")

    print("done")


if __name__ == "__main__":
    train_infer_net(problem_id="problem")
