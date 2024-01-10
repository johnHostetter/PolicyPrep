"""
This file is used to train the InferNet model for the problem level data.
"""
import argparse
import multiprocessing as mp
from typing import Tuple, List

from colorama import (
    init as colorama_init,
)  # for cross-platform colored text in the terminal
from colorama import Fore, Style  # for cross-platform colored text in the terminal

import torch
import numpy as np
import pandas as pd
from d3rlpy.dataset import MDPDataset
from skorch import NeuralNetRegressor
from skorch.callbacks import EarlyStopping
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_selection import SelectKBest
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import Normalizer, MinMaxScaler

from YACS.yacs import Config
from src.preprocess.data.parser import data_frame_to_d3rlpy_dataset

from src.preprocess.infernet.new_common import (
    read_data,
    build_model,
    calc_max_episode_length,
)
from src.utilities.reproducibility import (
    load_configuration,
    set_random_seed,
    path_to_project_root,
)

pd.options.mode.chained_assignment = None  # default='warn'


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
    is_problem_level, mdp_dataset, original_data = infernet_setup(problem_id)
    max_len = calc_max_episode_length(mdp_dataset)

    # select the features and actions depending on if the data is problem-level or step-level
    state_features, possible_actions = get_features_and_actions(
        config, is_problem_level
    )

    print(f"{problem_id}: Max episode length is {max_len}")

    NUM_OF_SELECTED_FEATURES = 30
    model, optimizer = build_model(
        NUM_OF_SELECTED_FEATURES + len(possible_actions) + 1  # + 1 for no-op action
    )

    class InferNet(NeuralNetRegressor):
        def __init__(self, module, *args, criterion=torch.nn.MSELoss, **kwargs):
            super(InferNet, self).__init__(module, *args, criterion=criterion, **kwargs)

        def get_loss(self, y_pred, y_true, X=None, training=False):
            from skorch.utils import to_tensor

            y_true = to_tensor(y_true, device=self.device)
            # print(y_true.sum(dim=1))
            return self.criterion_(
                y_pred.sum(1).flatten().to(self.device), y_true.sum(dim=1)
            )

    # create the skorch wrapper
    nn_reg = InferNet(
        model,
        criterion=torch.nn.MSELoss,
        optimizer=torch.optim.Adam,
        lr=3e-5,
        max_epochs=1000,
        batch_size=64,
        # train_split=predefined_split(val_data),
        device="cuda" if torch.cuda.is_available() else "cpu",
        callbacks=[
            EarlyStopping(patience=10, monitor="valid_loss"),
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

    # data preprocessing pipeline (model has to be separate due to data reshaping)
    pipeline = Pipeline(
        [
            ("flatten", Flatten()),
            (
                "scale",
                FeatureUnion(
                    [
                        ("minmax", MinMaxScaler()),
                        ("normalize", Normalizer()),
                    ]
                ),
            ),
            (
                "select",
                SelectKBest(k=NUM_OF_SELECTED_FEATURES),
            ),  # keep input size constant
            (
                "resize",
                MyReshape(
                    shape=(
                        len(mdp_dataset.episodes),
                        (max_len + 1),
                        NUM_OF_SELECTED_FEATURES,
                    )
                ),
            ),
        ]
    )

    pipeline.fit(mdp_dataset.observations, mdp_dataset.rewards)

    one_hot_encoded_actions = (
        OneHotEncoder()
        .fit_transform(mdp_dataset.actions.reshape(-1, 1))
        .toarray()
        .reshape(len(mdp_dataset.episodes), (max_len + 1), -1)
    )

    # too many zeros, so we only train on the non-zero rewards
    # indices = np.where(
    #     mdp_dataset.rewards.reshape(len(mdp_dataset.episodes), (max_len + 1))[:, -1]
    #     != 0.0
    # )[0]
    nn_reg.fit(
        np.concatenate(
            [
                pipeline.transform(mdp_dataset.observations)[
                    :, :-1, :
                ],  # remove the last step
                one_hot_encoded_actions[:, :-1, :],  # remove the last step
            ],
            axis=-1,
        ).astype(
            np.float32
        ),  # [indices],
        mdp_dataset.rewards.reshape(len(mdp_dataset.episodes), (max_len + 1))  # [
        # indices
        # ],  # don't remove the last step here, it contains the delayed reward
    )

    history_df = pd.DataFrame(nn_reg.history)
    # del 'batches' column (it is ugly)
    del history_df["batches"]

    iteration = 0  # this is expected but no longer needed

    # get the predictions
    predictions = nn_reg.predict(
        np.concatenate(
            [
                pipeline.transform(mdp_dataset.observations)[
                    :, :-1, :
                ],  # remove the last step
                one_hot_encoded_actions[:, :-1, :],  # remove the last step
            ],
            axis=-1,
        ).astype(np.float32)
    )
    results_df = original_data[
        ~original_data.reward.notna()
    ]  # remove the rows w/ delayed reward
    assert results_df.shape[0] == predictions.reshape(-1, 1).shape[0]
    results_df["reward"] = predictions.reshape(-1, 1)

    # save the predictions
    output_directory = path_to_project_root() / "data" / "with_inferred_rewards"
    output_directory.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_directory / f"{problem_id}_{iteration}.csv", index=False)

    # make the directory to save the model and loss data
    path_to_models = path_to_project_root() / "models" / "infernet"
    path_to_models.mkdir(parents=True, exist_ok=True)

    # save the history (e.g., loss data)
    history_df.to_csv(
        path_to_models / f"loss_{problem_id}_{iteration}.csv", index=False
    )
    # save the model
    torch.save(model, path_to_models / f"{problem_id}_{iteration}.pt")
    # save a checkpoint, to restore model weights and optimizer settings if training fails
    path_to_checkpoints = path_to_project_root() / "models" / "infernet" / "checkpoints"
    path_to_checkpoints.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model": model.state_dict(), "optimizer": optimizer.state_dict()},
        path_to_checkpoints / f"{problem_id}_{iteration}.pt",
    )  # we can use a dictionary to save any arbitrary information (e.g., lr_scheduler)
    print(
        f"{Fore.GREEN}"
        f"Saved model to {path_to_models / f'{problem_id}_{iteration}.pt'}"
        f"{Style.RESET_ALL}"
    )


def infernet_setup(problem_id: str) -> Tuple[bool, MDPDataset, pd.DataFrame]:
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
    # the following function will create a MDPDataset as well as further clean the data once more
    mdp_dataset, original_data = data_frame_to_d3rlpy_dataset(original_data, problem_id)
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
        print(
            f"{Fore.YELLOW}"
            f"Training the InferNet model for {problem_id}..."
            f"{Style.RESET_ALL}"
        )
        if problem_id not in config.training.skip.problems:
            train_infer_net(f"{problem_id}(w)")
            # pool.apply_async(train_infer_net, args=(f"{problem_id}(w)",))
        # pool.close()
        # pool.join()
    print(
        f"{Fore.GREEN}"
        "All processes finished for training step-level InferNet models."
        f"{Style.RESET_ALL}"
    )


if __name__ == "__main__":
    train_infer_net(problem_id="problem")
