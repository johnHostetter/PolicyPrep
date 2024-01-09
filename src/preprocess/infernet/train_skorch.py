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
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import SelectKBest
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import Normalizer, MinMaxScaler
from skorch import NeuralNetRegressor
from skorch.callbacks import EarlyStopping

from YACS.yacs import Config
from soft.computing.blueprints.factory import SystematicDesignProcess
from soft.computing.wrappers.supervised import SupervisedDataset
from soft.fuzzy.logic.controller import Specifications
from soft.fuzzy.logic.controller.impl import ZeroOrderTSK, Mamdani
from soft.computing.wrappers.temporal import TimeDistributed
from src.preprocess.data.parser import data_frame_to_d3rlpy_dataset

from src.preprocess.infernet.new_common import (
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
    is_problem_level, mdp_dataset, original_data = infernet_setup(problem_id, config=config)
    max_len = calc_max_episode_length(mdp_dataset)
    user_ids: List[int] = original_data["userID"].unique().tolist()

    # select the features and actions depending on if the data is problem-level or step-level
    state_features, possible_actions = get_features_and_actions(
        config, is_problem_level
    )

    # create the buffer to train InferNet from
    infer_buffer = create_buffer(
        original_data,
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
        soft_config.fuzzy.partition.adjustment = 0.2
        soft_config.fuzzy.partition.epsilon = 0.7

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

    knowledge_base = None
    self_organize = SystematicDesignProcess(
        algorithms=["expert_partition", "no_rules"], config=soft_config
    ).build(
        training_data=SupervisedDataset(inputs=train_transitions[:10000], targets=None),
        # validation_data=val_data,
    )
    knowledge_base = self_organize.start()

    # flc = ZeroOrderTSK(
    #     1,
    #     knowledge_base=knowledge_base,
    #     input_trainable=False,
    #     consequences=None,
    # )


    NUM_OF_SELECTED_FEATURES = 30
    model, optimizer = build_model(NUM_OF_SELECTED_FEATURES + len(possible_actions) + 1)
    # model, optimizer = build_model(len(state_features))

    specifications = Specifications(
        t_norm="algebraic_product",
        number_of_variables={
            # "input": len(state_features),
            "input": NUM_OF_SELECTED_FEATURES,
            "output": 1,
            # "output": len(possible_actions),
        },
        number_of_terms={
            "input": 5,
            "output": 1,
        },
        expandable_variables={
            "input": True,
            "output": False,
        },
        number_of_rules=25,
    )
    # knowledge_base = None

    # FLC from network morphism
    flc = ZeroOrderTSK(
        specifications=specifications, knowledge_base=None,
        # disabled_parameters=["centers", "widths"]
    )

    # infer_net = torch.nn.Sequential(
    #     flc,
    #     torch.nn.LazyLinear(out_features=128),
    #     torch.nn.LeakyReLU(),
    #     # torch.nn.LazyLinear(out_features=32),
    #     # torch.nn.LeakyReLU(),
    #     torch.nn.LazyLinear(out_features=3),
    # )
    infer_net = flc

    # model = TimeDistributed(module=infer_net, batch_first=True).cuda()
    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-1)

    # Train the InferNet model
    print("#####################")

    class InferNet(NeuralNetRegressor):
        def __init__(
            self,
            module,
            *args,
            criterion=torch.nn.MSELoss,
            **kwargs
        ):
            super(InferNet, self).__init__(
                module,
                *args,
                criterion=criterion,
                **kwargs
            )

        def get_loss(self, y_pred, y_true, X=None, training=False):
            from skorch.utils import to_tensor
            y_true = to_tensor(y_true, device=self.device)
            # print(y_true.sum(dim=1))
            return self.criterion_(y_pred.sum(1).flatten().to(self.device), y_true.sum(dim=1))


    # create the skorch wrapper
    nn_reg = InferNet(
        # Net(len(input_features), 1),
        model,
        criterion=torch.nn.MSELoss,
        optimizer=torch.optim.Adam,
        lr=1e-5,
        max_epochs=1000,
        batch_size=4,
        # train_split=predefined_split(val_data),
        device="cuda" if torch.cuda.is_available() else "cpu",
        callbacks=[
            EarlyStopping(patience=10, monitor="valid_loss"),
            # UpdateOptimizerCallback(),
        ],
    )

    class Flatten(BaseEstimator, TransformerMixin):
        def __init__(self):
            pass

        def fit(self, X, y=None):
            return self

        def transform(self, X):
            return X.reshape(-1, X.shape[-1])

    class MyReshape(BaseEstimator, TransformerMixin):
        def __init__(self, shape):
            self.shape = shape

        def fit(self, X, y=None):
            return self

        def transform(self, X):
            return X.reshape(self.shape)

    pipeline = Pipeline(
        [
            ("flatten", Flatten()),
            (
                "scale",
                # Normalizer(),  # MinMaxScaler(),
                FeatureUnion(
                    [
                        ("minmax", MinMaxScaler()),
                        ("normalize", Normalizer()),
                    ]
                ),
            ),
            ("select", SelectKBest(k=NUM_OF_SELECTED_FEATURES)),  # keep input size constant
            ("resize", MyReshape(shape=(len(mdp_dataset.episodes), (max_len + 1), NUM_OF_SELECTED_FEATURES))),
            # ("net", nn_reg),
        ]
    )

    pipeline.fit(
        mdp_dataset.observations,  #.reshape(len(mdp_dataset.episodes), (max_len + 1), len(state_features)),
        mdp_dataset.rewards  #.reshape(len(mdp_dataset.episodes), (max_len + 1))
    )

    from sklearn.preprocessing import OneHotEncoder

    one_hot_encoded_actions = OneHotEncoder().fit_transform(mdp_dataset.actions.reshape(-1, 1)).toarray().reshape(len(mdp_dataset.episodes), (max_len + 1), -1)

    # one_hot_encoded_actions = np.zeros((len(mdp_dataset.episodes) * (max_len + 1), len(possible_actions)))
    # one_hot_encoded_actions[np.arange(mdp_dataset.actions.size), mdp_dataset.actions] = 1
    indices = np.where(mdp_dataset.rewards.reshape(len(mdp_dataset.episodes), (max_len + 1))[:, -1] != 0.0)[0]
    nn_reg.fit(
        np.concatenate([pipeline.transform(mdp_dataset.observations), one_hot_encoded_actions], axis=-1).astype(np.float32)[indices],
        mdp_dataset.rewards.reshape(len(mdp_dataset.episodes), (max_len + 1))[indices]
    )





    start_time = time.time()
    losses = []
    criterion = torch.nn.HuberLoss()
    # temp = 5.0
    for iteration in range(config.training.data.num_iterations + 1):
        # if iteration < 810000:
        #     continue
        # elif iteration == 810000:
        #     checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
        #     path_to_checkpoints = path_to_project_root() / "models" / "infernet" / "checkpoints"
        #     checkpoint.restore(path_to_checkpoints / f"{problem_id}_{iteration}-1")
        # else:
        batch_size = 1
        batch = random.sample(infer_buffer, batch_size)
        states_actions, _, _, imm_rew_sum, _ = list(zip(*batch))
        states_actions = np.reshape(
            states_actions,
            (batch_size, max_len, num_state_and_actions),
        )
        states = states_actions[:, :, :-len(possible_actions)]
        actions = states_actions[:, :, -len(possible_actions):]
        imm_rew_sum = np.reshape(imm_rew_sum, (batch_size, 1))
        model.actions = actions
        predicted_immediate_rewards = model.cuda()(torch.Tensor(states_actions[:, :, :-3]).cuda()).cpu()
        loss = criterion(predicted_immediate_rewards, torch.Tensor(imm_rew_sum).cpu())
        # predicted_immediate_rewards = model(torch.Tensor(states_actions[:, :, np.array(mask.tolist() + [True, True, True])]))
        # selected_immediate_rewards = predicted_immediate_rewards.gather(
        #     -1, torch.argmax(torch.tensor(actions), dim=-1, keepdim=True)
        # ).cpu()
        # # loss = criterion(predicted_immediate_rewards.sum(dim=1), torch.Tensor(imm_rew_sum).cpu())
        # loss = criterion(selected_immediate_rewards.sum(dim=1), torch.Tensor(imm_rew_sum).cpu())
        optimizer.zero_grad()
        loss.backward()
        # if iteration > 0 and iteration % 1000 == 0:
        #     temp = model.module.engine.intermediate_calculation_modules.temperature.item()
        #     val = np.mean(losses[-2000:-1000]) - np.mean(losses[-1000:])
        #     if np.isnan(val):
        #         val = 0.0
        #     temp -= val
        #     temp -= 0.1
        #     temp = max(1e-2, temp)
        #     print(val)
        #     model.module.engine.intermediate_calculation_modules.temperature = torch.nn.Parameter(torch.Tensor([temp]).cuda())
        if iteration > 0 and iteration % 1000 == 0:
            print(iteration, np.mean(losses[-1000:]))
            # # temp = model.module.engine.temperature.item()
            # temp = model.module.engine.intermediate_calculation_modules.temperature.item()
            # # temp = max(1e-2, model.module[0].engine.intermediate_calculation_modules.temperature.item())
            # print(iteration, loss.item(), model.module.engine.intermediate_calculation_modules.temperature)
        # print(loss.item())
        losses.append(loss.item())
        if False and iteration > 0 and iteration % config.training.data.checkpoint == 0:
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
    # normalized_data = normalize_data(
    #     original_data, problem_id, columns_to_normalize=state_features
    # )
    mdp_dataset = data_frame_to_d3rlpy_dataset(original_data, problem_id)
    return is_problem_level, mdp_dataset, original_data


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
