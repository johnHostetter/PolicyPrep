"""
Use the DQN algorithm from d3rlpy to train a policy.
"""
from typing import List
import multiprocessing as mp

import torch
import d3rlpy.algos
import numpy as np
import pandas as pd
from colorama import (
    init as colorama_init,
)  # for cross-platform colored text in the terminal
from colorama import Fore, Style  # for cross-platform colored text in the terminal
from sklearn.model_selection import train_test_split

from d3rlpy.dataset import MDPDataset, Episode
from d3rlpy.metrics import td_error_scorer, average_value_estimation_scorer

from YACS.yacs import Config
from src.policy.induction.d3rlpy.nas import modify_algorithm
from src.utilities.reproducibility import load_configuration, path_to_project_root
# the following are imports required for a closed-sourced library called PySoft
# if you don't have PySoft, you need to comment out the following imports
# and the code that uses them
from soft.computing.wrappers.old_d3rlpy import (
    CustomEncoderFactory as SoftEncoderFactory,
    CustomMeanQFunctionFactory
)
from soft.computing.wrappers.supervised import SupervisedDataset
from soft.computing.blueprints.factory import SystematicDesignProcess

colorama_init()  # initialize the colorama module


def induce_policies_with_d3rlpy(num_workers: int = 1) -> None:
    """
    Iterate over the different semesters of training data and induce policies using the DQN
    algorithm.

    Args:
        num_workers: The number of processes to use for multiprocessing.

    Returns:
        None
    """
    # load the configuration settings
    pd.options.mode.chained_assignment = None
    config = load_configuration("default_configuration.yaml")

    problems = list(config.training.problems)
    problems.insert(0, "problem")

    # with mp.Pool(processes=num_workers) as pool:
    path_to_policy_data_directory = (
        path_to_project_root()
        / "data"
        / "for_policy_induction"
        / config.training.data.policy
    )
    for algo in config.training.algorithms:
        for problem_id in problems:
            if problem_id in config.training.skip.problems:
                continue
            if "problem" not in problem_id:
                problem_id += "(w)"
            try:
                path_to_data = (
                    path_to_policy_data_directory / "d3rlpy" / f"{problem_id}.h5"
                )
                if not path_to_data.exists():
                    print(
                        f"{Fore.RED}"
                        f"Skipping {path_to_data.name} (it does not exist)..."
                        f"{Style.RESET_ALL}"
                    )
                    continue
                if path_to_data.is_dir():
                    print(
                        f"{Fore.RED}"
                        f"Skipping {path_to_data.name} (it is a directory)..."
                        f"{Style.RESET_ALL}"
                    )
                    continue
                print(
                    f"{Fore.YELLOW}"
                    f"Using data from {path_to_data.name}..."
                    f"{Style.RESET_ALL}"
                )
                mdp_dataset = MDPDataset.load(str(path_to_data))
                print(
                    f"{Fore.YELLOW}"
                    f"Using {len(mdp_dataset.episodes)} episodes of data to induce a policy "
                    f"for {path_to_data.name}..."
                    f"{Style.RESET_ALL}"
                )

            except FileNotFoundError as file_not_found_error:
                print(repr(file_not_found_error))
                continue

            train_d3rlpy_policy(
                mdp_dataset, problem_id, d3rlpy.algos.get_algo(algo, discrete=True), config
            )

        #         x = pool.apply_async(
        #             train_d3rlpy_policy,
        #             args=[
        #                 mdp_dataset,
        #                 problem_id,
        #                 d3rlpy.algos.get_algo(algo, discrete=True),
        #                 config,
        #             ],
        #         )
        #         x.get()  # check if any Exception has been thrown and display traceback
        # pool.close()
        # pool.join()


def train_d3rlpy_policy(
    mdp_dataset: MDPDataset, problem_id: str, d3rlpy_alg, config: Config
) -> None:
    """
    Train a DQN policy.

    Args:
        mdp_dataset: The MDP dataset for the specified problem ID.
        problem_id: The ID of the problem, or "problem" if the dataset is "problem-level".
        d3rlpy_alg: The selected algorithm from the library called d3rlpy.
        config: The configuration settings.

    Returns:
        None
    """
    train_episodes, test_episodes = train_test_split(mdp_dataset, test_size=0.2)
    n_state = train_episodes[0].observations.shape[-1]
    knowledge_base = None  # this is only used for PySoft

    ################################################################################################
    #              The following code is to create a neuro-fuzzy network with PySoft.              #
    ################################################################################################

    # train_episodes = train_episodes[:10]
    # test_episodes = test_episodes[:10]

    # import soft.computing.blueprints

    soft_config = load_configuration(
        path_to_project_root()
        / "PySoft"
        / "configurations"
        / "default_configuration.yaml"
    )

    # # change default device for PyTorch to use the GPU if it is available
    device = "cpu"
    # torch.set_default_device("cpu")
    with soft_config.unfreeze():
        soft_config.device = device  # reflect those changes in the Config object
        if soft_config.fuzzy.t_norm.yager.lower() == "euler":
            w_parameter = np.e
        elif soft_config.fuzzy.t_norm.yager.lower() == "golden":
            w_parameter = (1 + 5**0.5) / 2
        else:
            w_parameter = float(soft_config.fuzzy.t_norm.yager)
        soft_config.fuzzy.t_norm.yager = w_parameter
        soft_config.fuzzy.rough.compatibility = False
        soft_config.output.name = (
            path_to_project_root()
            / "figures"
            / "CEW"
            / d3rlpy_alg.__name__
            / problem_id
        )
        soft_config.clustering.distance_threshold = 0.17
        soft_config.training.epochs = 300
        soft_config.training.learning_rate = 3e-4
        soft_config.fuzzy.t_norm.yager = np.e
        soft_config.fuzzy.partition.adjustment = 0.2
        soft_config.fuzzy.partition.epsilon = 0.7

    train_transitions = torch.Tensor(
        np.array(
            [
                transition
                for episode in train_episodes
                for transition in episode.observations
            ]
        )  # outer loop is first, then inner loop
    )
    test_transitions = np.array(
        [transition for episode in test_episodes for transition in episode.observations]
    )  # outer loop is first, then inner loop
    # FLC from self-organizing
    self_organize = SystematicDesignProcess(
        # algorithms=["clip", "no_rules"], config=soft_config
        algorithms=["expert_partition", "no_rules"],
        config=soft_config,
    ).build(
        training_data=SupervisedDataset(inputs=train_transitions, targets=None),
        validation_data=SupervisedDataset(inputs=test_transitions, targets=None),
    )
    # knowledge_base = self_organize.start()
    # min_values, _ = train_transitions.min(axis=0)
    # max_values, _ = train_transitions.max(axis=0)
    # mask = min_values != max_values

    # print(f"{problem_id}: {np.array(config.data.features.step)[min_values == max_values]}")
    # return

    # train_transitions = torch.tensor(train_transitions[:, mask])
    # test_transitions = torch.tensor(test_transitions[:, mask])
    # self_organize = soft.computing.blueprints.clip_ecm_wm(
    #     train_transitions,
    #     # test_transitions,
    #     config=soft_config
    # )
    #
    # n_state = train_transitions.shape[-1]
    # torch.cuda.empty_cache()
    #
    # knowledge_base = self_organize.start()
    # # path_to_kb_dirs = path_to_project_root() / "models" / "policies" / d3rlpy_alg.__name__ / "pysoft" / problem_id
    # # path_to_kb = list(path_to_kb_dirs.glob("*"))[-1]  # get the last entry
    # # from soft.computing.knowledge import KnowledgeBase
    # # knowledge_base = KnowledgeBase.load(path_to_kb)
    # #
    # # # update the config for the knowledge_base
    # # knowledge_base.config = load_configuration()  # just load default
    # fuzzy_rules = knowledge_base.get_fuzzy_logic_rules()
    # premises = [len(fuzzy_rule.premise) for fuzzy_rule in fuzzy_rules]
    # print(
    #     f"Solving the system with "
    #     f"{len(fuzzy_rules)} rules."
    # )
    # print(
    #     f"Average (std. dev.) premises is "
    #     f"{np.mean(premises)} ({np.std(premises)})"
    # )
    if problem_id == "problem":
        n_action = len(config.training.actions.problem)
    else:
        n_action = len(config.training.actions.step)

    # with soft_config.unfreeze():
    #     soft_config.device = torch.device(
    #         f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu"
    #     )
    #     torch.set_default_device(soft_config.device)
    #
    # knowledge_base.config = soft_config

    encoder_factory = SoftEncoderFactory(
        knowledge_base=knowledge_base, feature_size=n_state, action_size=n_action
    )

    ################################################################################################
    #               The following code is to induce a policy with the d3rlpy library.              #
    ################################################################################################

    print(
        f"{Fore.YELLOW}"
        f"Training a {d3rlpy_alg.__name__} policy for {problem_id}..."
        f"{Style.RESET_ALL}"
    )
    algorithm = d3rlpy_alg(
        # remove encoder_factory=encoder_factory, to use the default encoder
        encoder_factory=encoder_factory,
        # remove q_func_factory=CustomMeanQFunctionFactory(share_encoder=True), to use the default Q-function
        q_func_factory=CustomMeanQFunctionFactory(share_encoder=True),
        # keep the following parameters as-is (or change them if you want to experiment)
        use_gpu=torch.cuda.is_available(),
        learning_rate=1e-3,
        # learning_rate=1e-4,
        batch_size=64,
        # alpha=0.1,
    )

    def make_terminals(episodes: List[Episode]) -> List[int]:
        """
        The d3rlpy version used at this time does not have a 'terminals' member.
        This method calculates the correct 'terminals' list.

        Args:
            episodes: A list of episodes.

        Returns:
            A list that contains zeros or ones, zero means that the corresponding transition/state
            is not a terminal state (i.e., the end of the episode has not yet been reached), whereas
            a one denotes that the end of an episode has been reached.
        """
        terminals = []
        for episode in episodes:
            terminals.extend([0] * len(episode))
            terminals[-1] = 1
        return terminals

    path_to_logs = (
        path_to_project_root()
        / "logs"
        / config.training.data.policy
        / d3rlpy_alg.__name__
        / problem_id
    )
    path_to_logs.mkdir(parents=True, exist_ok=True)
    # modify the algorithm instance to override the handling of the optimizer for NAS
    algorithm = modify_algorithm(algorithm, n_state, n_action)

    algorithm.fit(
        train_episodes,
        eval_episodes=test_episodes,
        save_interval=350,  # save the model every 500 epochs
        n_epochs=32,
        logdir=str(path_to_logs),
        scorers={
            "td_error": td_error_scorer,
            "value_scale": average_value_estimation_scorer,
        },
    )
    # algorithm.fit(
    #     MDPDataset(
    #         train_transitions.detach().numpy(),
    #         np.array([action for episode in train_episodes for action in episode.actions]),
    #         np.array([reward for episode in train_episodes for reward in episode.rewards]),
    #         np.array(make_terminals(train_episodes)),
    #     ),
    #     eval_episodes=MDPDataset(
    #         test_transitions.detach().numpy(),
    #         np.array([action for episode in test_episodes for action in episode.actions]),
    #         np.array([reward for episode in test_episodes for reward in episode.rewards]),
    #         np.array(make_terminals(test_episodes)),
    #     ),
    #     n_epochs=100,
    #     logdir=str(path_to_logs),
    #     scorers={
    #         "td_error": td_error_scorer,
    #         "value_scale": average_value_estimation_scorer,
    #     },
    # )
    print(
        f"{Fore.GREEN}"
        f"Finished training a {d3rlpy_alg.__name__} policy for {problem_id}..."
        f"{Style.RESET_ALL}"
    )

    # from d3rlpy.ope import DiscreteFQE
    #
    # fqe = DiscreteFQE(algo=algorithm)
    # from d3rlpy.metrics.scorer import initial_state_value_estimation_scorer
    # from d3rlpy.metrics.scorer import soft_opc_scorer
    #
    # fqe.fit(train_episodes, eval_episodes=test_episodes, n_epochs=12,
    #         scorers={
    #             "init_value": initial_state_value_estimation_scorer,
    #             "soft_opc": soft_opc_scorer(return_threshold=600)
    #         })

    # make the directory if it does not exist already and save the model
    print(
        f"{Fore.YELLOW}"
        f"Saving the {d3rlpy_alg.__name__} policy for {problem_id}..."
        f"{Style.RESET_ALL}"
    )
    for directory in ["d3rlpy", "trace", "pysoft", "onnx"]:
        output_directory = (
            path_to_project_root() / "models" / "policies" / config.training.data.policy
        )
        output_directory = output_directory / d3rlpy_alg.__name__ / directory
        output_directory.mkdir(parents=True, exist_ok=True)
        if directory == "d3rlpy":
            # save the algorithm model to the output directory
            algorithm.save_model(str(output_directory / f"{problem_id}.pt"))
            print(
                f"{Fore.GREEN}"
                f"Finished saving the {d3rlpy_alg.__name__} policy "
                f"for {problem_id} as a d3rlpy model..."
                f"{Style.RESET_ALL}"
            )
        elif directory == "trace":
            # # save the traced model to the output directory (for use in the web app)
            # max_length = max([len(episode.observations) for episode in train_episodes])
            # train_transitions = torch.tensor(np.array([
            #     torch.nn.functional.pad(
            #         torch.tensor(episode.observations),
            #         pad=(0, 0, 0, max_length - episode.observations.shape[0])
            #     ).numpy() for episode in train_episodes
            # ]))
            # train_transitions = train_transitions.view(-1, train_transitions.shape[-1]).cpu()
            # # fetch the function approximator that is used for the Q-function
            # model = algorithm.impl.q_function._q_funcs[0]._encoder
            # # enable the evaluation mode for the model to be traced
            # model.eval()  # torch.nn.Parameters cannot be traced in training mode
            # model.cpu()  # the traced model will be used in the web app, which is CPU-only
            # traced_model = torch.jit.trace(
            #     model, train_transitions
            # )
            # torch.jit.save(traced_model, str(output_directory / f"{problem_id}.pt"))

            # save greedy-policy as TorchScript
            algorithm.save_policy(str(output_directory / f"{problem_id}.pt"))
            print(
                f"{Fore.GREEN}"
                f"Saved the {d3rlpy_alg.__name__} policy for {problem_id} as a TorchScript model..."
                f"{Style.RESET_ALL}"
            )
        elif directory == "pysoft":
            if knowledge_base is not None:
                knowledge_base.save(str(output_directory / f"{problem_id}"))
                print(
                    f"{Fore.GREEN}"
                    f"Saved the {d3rlpy_alg.__name__} policy "
                    f"for {problem_id} as a PySoft KnowledgeBase..."
                    f"{Style.RESET_ALL}"
                )
            else:
                # no need for the pysoft subdirectory - delete the directory if it exists & is empty
                if (
                    output_directory.exists()
                    and output_directory.is_dir()
                    and not any(output_directory.iterdir())
                ):
                    output_directory.rmdir()
        else:
            # save greedy-policy as ONNX
            if torch.cuda.is_available():
                train_transitions = train_transitions.cuda()
            # algorithm.save_policy(
            #     str(output_directory / f"{problem_id}.onnx")
            # )  # this command should have worked, but it did not, so we use the following
            model = algorithm._impl._q_func.q_funcs[
                0
            ].cpu()  # do not use CUDA when exporting, it can cause issues
            model.eval()  # torch.nn.Parameters cannot & should not be traced in training mode
            torch.onnx.export(
                model,  # the trained Q-function is in the 0'th index
                torch.randn(1, n_state),
                output_directory / f"{problem_id}.onnx",
                opset_version=11,
                input_names=["x"],
                output_names=["y"],
            )
            print(
                f"{Fore.GREEN}"
                f"Saved the {d3rlpy_alg.__name__} policy for {problem_id} as an ONNX model..."
                f"{Style.RESET_ALL}"
            )
    print(
        f"{Fore.GREEN}"
        f"Finished saving all files of the {d3rlpy_alg.__name__} policy for {problem_id}."
        f"{Style.RESET_ALL}"
    )


if __name__ == "__main__":
    induce_policies_with_d3rlpy()
