"""
This file is used to train the InferNet model for the step level data.
"""
import os
import time
import random

import numpy as np
import pandas as pd
import tensorflow as tf

from src.utils.reproducibility import load_configuration, set_random_seed

from src.preprocess.infernet.common import (
    read_data,
    model_build,
    calc_max_episode_length,
    normalize_data,
    create_buffer,
    infer_and_save_rewards,
)

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
config = load_configuration()
set_random_seed(seed=config.training.seed)


def main(file_name):
    tf.keras.backend.set_floatx("float64")
    original_data = read_data(file_name)
    user_ids = original_data["userID"].unique()
    max_len = calc_max_episode_length(original_data, user_ids, config)

    num_state_features = len(config.data.features.problem)
    num_actions = len(config.training.actions.problem)
    num_state_and_actions = num_state_features + num_actions
    print(file_name)
    print(f"Max episode length is {max_len}")

    normalized_data = normalize_data(
        original_data, file_name, columns_to_normalize=config.data.features.step
    )

    # Train Infer Net.
    model = model_build(max_len, num_state_features + num_actions)
    infer_buffer = create_buffer(
        normalized_data, user_ids, config, is_problem_level=False, max_len=max_len
    )

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
            print(file_name)
        if iteration % 1000 == 0:
            print(f"Step {iteration}/{train_steps}, loss {loss}")
            print("Training time is", time.time() - start_time, "seconds")
            start_time = time.time()

        if iteration in (5000, 10000):
            # Infer the rewards for the data and save the data.
            infer_and_save_rewards(
                file_name,
                iteration,
                infer_buffer,
                max_len,
                model,
                normalized_data,
                num_state_and_actions,
                is_problem_level=False,
            )

            df = pd.DataFrame({"loss": losses})
            df.to_csv(f"figures/loss_{file_name}_{iteration}.csv", index=False)
            model.save(f"model/model_{file_name}_{iteration}.h5")

    print("done")


if __name__ == "__main__":
    # file_names = ['features_all_ex132', 'features_all_ex132a', 'features_all_ex152',
    #               'features_all_ex152a', 'features_all_ex152b', 'features_all_ex212',
    #               'features_all_ex242', 'features_all_ex252', 'features_all_ex252a'
    #               ]
    #
    # for file_name in file_names:
    file_name = "features_all_exc137"
    main(file_name)
    # main(sys.argv[1])
