"""
This file is used to train the InferNet model for the problem level data.
"""
import os
import time
import random
import argparse
import multiprocessing as mp
from typing import Tuple, List

import torch
import numpy as np
import pandas as pd
from d3rlpy.dataset import MDPDataset
from skorch import NeuralNetRegressor

from YACS.yacs import Config
from soft.fuzzy.logic.controller import Specifications
from soft.fuzzy.logic.controller.impl import ZeroOrderTSK
from soft.computing.wrappers.temporal import TimeDistributed
from src.preprocess.data.parser import data_frame_to_d3rlpy_dataset

from src.preprocess.infernet.common import (
    read_data,
    build_model,
    calc_max_episode_length,
    normalize_data,
    create_buffer,
    infer_and_save_rewards,
)
from src.utilities.reproducibility import (
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
    # load the configuration file
    config = load_configuration()
    # set the random seed
    set_random_seed(seed=config.training.seed)
    is_problem_level, mdp_dataset, normalized_data = infernet_setup(problem_id, config=config)
    max_len = calc_max_episode_length(mdp_dataset)
    user_ids: List[int] = normalized_data["userID"].unique().tolist()

    # select the features and actions depending on if the data is problem-level or step-level
    state_features, possible_actions = get_features_and_actions(
        config, is_problem_level
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

    # PySoft
    # import soft.computing.blueprints
    # from soft.computing.wrappers.d3rlpy import CustomEncoderFactory as SoftEncoderFactory

    soft_config = load_configuration(
        path_to_project_root() / "PySoft" / "configurations" / "default_configuration.yaml"
    )

    with soft_config.unfreeze():
        # soft_config.device = device  # reflect those changes in the Config object
        if soft_config.fuzzy.t_norm.yager.lower() == "euler":
            w_parameter = np.e
        elif soft_config.fuzzy.t_norm.yager.lower() == "golden":
            w_parameter = (1 + 5 ** 0.5) / 2
        else:
            w_parameter = float(soft_config.fuzzy.t_norm.yager)
        soft_config.fuzzy.t_norm.yager = w_parameter
        soft_config.fuzzy.rough.compatibility = False
        # soft_config.output.name = path_to_project_root() / "figures" / "CEW" / d3rlpy_alg.__name__ / problem_id
        # soft_config.clustering.distance_threshold = 0.17
        # soft_config.training.epochs = 300
        soft_config.training.learning_rate = 3e-4
        # soft_config.fuzzy.t_norm.yager = np.e
        # soft_config.fuzzy_antecedents_generation.alpha = 0.2
        # soft_config.fuzzy_antecedents_generation.beta = 0.7

    train_transitions = np.array(
        [transition for episode in mdp_dataset.episodes for transition in episode.observations]
    )  # outer loop is first, then inner loop
    # test_transitions = np.array(
    #     [transition for episode in test_episodes for transition in episode.observations]
    # )  # outer loop is first, then inner loop
    min_values = train_transitions.min(axis=0)
    max_values = train_transitions.max(axis=0)
    mask = min_values != max_values
    train_transitions = torch.tensor(train_transitions[:, mask])
    # test_transitions = torch.tensor(test_transitions[:, mask])

    # self_organize = soft.computing.blueprints.clip_ecm_wm(
    #     train_transitions[:10000],
    #     # test_transitions,
    #     config=soft_config
    # )

    # knowledge_base = self_organize.start()

    # flc = ZeroOrderTSK(
    #     1,
    #     knowledge_base=knowledge_base,
    #     input_trainable=False,
    #     consequences=None,
    # )



    # model, optimizer = build_model(len(state_features) + len(possible_actions))

    specifications = Specifications(
        t_norm="algebraic_product",
        number_of_variables={
            "input": len(state_features),
            "output": 1,
        },
        number_of_terms={
            "input": 5,
            "output": 1,
        },
        expandable_variables={
            "input": True,
            "output": False,
        },
        number_of_rules=100,
    )
    knowledge_base = None

    # FLC from network morphism
    flc = ZeroOrderTSK(
        specifications=specifications, knowledge_base=knowledge_base
    )

    model = TimeDistributed(module=flc, batch_first=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Train the InferNet model
    print("#####################")
    start_time = time.time()
    losses = []
    criterion = torch.nn.MSELoss()
    for iteration in range(config.training.data.num_iterations + 1):
        # if iteration < 810000:
        #     continue
        # elif iteration == 810000:
        #     checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
        #     path_to_checkpoints = path_to_project_root() / "models" / "infernet" / "checkpoints"
        #     checkpoint.restore(path_to_checkpoints / f"{problem_id}_{iteration}-1")
        # else:
        batch = random.sample(infer_buffer, config.training.data.batch_size)
        states_actions, _, _, imm_rew_sum, _ = list(zip(*batch))
        states_actions = np.reshape(
            states_actions,
            (config.training.data.batch_size, max_len, num_state_and_actions),
        )
        imm_rew_sum = np.reshape(imm_rew_sum, (config.training.data.batch_size, 1))
        predicted_immediate_rewards = model(torch.Tensor(states_actions[:, :, :-3]).cuda()).cpu()
        # predicted_immediate_rewards = model(torch.Tensor(states_actions[:, :, np.array(mask.tolist() + [True, True, True])]))
        loss = criterion(predicted_immediate_rewards.sum(dim=1), torch.Tensor(imm_rew_sum).cpu())
        optimizer.zero_grad()
        loss.backward()
        if iteration > 0 and iteration % 100 == 0:
            print(iteration, loss.item())
        # print(loss.item())
        losses.append(loss.item())
        if iteration > 0 and iteration % config.training.data.checkpoint == 0:
            print(
                f"Step: {iteration}/{config.training.data.num_iterations}, "
                f"Loss: {np.mean(losses[-config.training.data.checkpoint:])}"
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
            torch.save(model, path_to_models / f"{problem_id}_{iteration}.pt")
            # save a checkpoint, to restore model weights and optimizer settings if training fails
            path_to_checkpoints = path_to_project_root() / "models" / "infernet" / "checkpoints"
            path_to_checkpoints.mkdir(parents=True, exist_ok=True)
            torch.save(
                {"model": model.state_dict(), "optimizer": optimizer.state_dict()},
                path_to_checkpoints / f"{problem_id}_{iteration}.pt"
            )  # we can use a dictionary to save any arbitrary information (e.g., lr_scheduler)
    print(f"Done training InferNet for {problem_id}.")


def infernet_setup(problem_id: str, config: Config) -> Tuple[bool, MDPDataset, pd.DataFrame]:
    """
    Set up the InferNet model for the problem level data if "problem" is in problem_id. Otherwise,
    set up the InferNet model for the step level data.

    Args:
        problem_id:
        config:

    Returns:
        A tuple containing the following:
            is_problem_level: A boolean indicating if the data is problem-level or step-level.
            mdp_dataset: The MDPDataset representation where the state features are normalized.
            normalized_data: The normalized dataset.
    """
    # determine if the data is problem-level or step-level
    is_problem_level = "problem" in problem_id
    original_data = read_data(problem_id, "for_inferring_rewards", selected_users=None)
    # select the features and actions depending on if the data is problem-level or step-level
    state_features, possible_actions = get_features_and_actions(
        config, is_problem_level
    )
    normalized_data = normalize_data(
        original_data, problem_id, columns_to_normalize=state_features
    )
    mdp_dataset = data_frame_to_d3rlpy_dataset(normalized_data, problem_id)
    return is_problem_level, mdp_dataset, normalized_data


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
    # with mp.Pool(processes=args.num_workers) as pool:
    for problem_id in config.training.problems:
        print(f"Training the InferNet model for {problem_id}...")
        if problem_id not in config.training.skip.problems:
            train_infer_net(f"{problem_id}(w)")
                # pool.apply_async(train_infer_net, args=(f"{problem_id}(w)",))
        # pool.close()
        # pool.join()
    print("All processes finished for training step-level InferNet models.")


if __name__ == "__main__":
    train_infer_net(problem_id="problem")
