"""
Evaluate the given policy on the selected problem ID; if problem_id is "problem", then evaluate
the given policy on the problem-level scenario. Otherwise, evaluate the given policy on the
step-level scenario. This module implements the offline off-policy evaluation (OPE) methods for
the two possible scenarios. The OPE methods are:
    - Inverse Propensity Scoring (IPS)
    - Doubly Robust (DR)
    - Weighted Doubly Robust (WDR)
    - Weighted Importance Sampling (WIS)
    - Per-decision Importance Sampling (PDIS)
    - Per-decision Weighted Importance Sampling (PWIS)
    - Per-decision Hybrid Weighted Importance Sampling (PHWIS)

The OPE methods are implemented in the ImportanceSampling class, which is imported from
src/utils/importance_sampling.py.

These metrics help give a sense of how well the policy performs on the given problem ID. The
metrics are computed by comparing the policy's actions to the real actions taken by the user.
The real actions are inferred from the user's interactions with the system.
"""
from typing import Dict, List

import numpy as np
import pandas as pd
from alive_progress import alive_bar
from pandas import DataFrame
from colorama import (
    init as colorama_init,
)  # for cross-platform colored text in the terminal
from colorama import Fore, Style  # for cross-platform colored text in the terminal

from YACS.yacs import Config
from src.utilities.reproducibility import path_to_project_root, load_configuration
from src.utilities.importance_sampling import ImportanceSampling

colorama_init(autoreset=True)  # for cross-platform colored text in the terminal


def evaluate_policy_with_importance_sampling(
    policy_name: str,
    problem_id: str,
    config: Config,
) -> DataFrame:
    """
    Evaluate the given policy on the selected problem ID; if problem_id is "problem",
    then evaluate the given policy on the problem-level scenario.

    Args:
        policy_name: The name of the policy.
        problem_id: The problem ID.
        config: The configuration settings.

    Returns:
        A DataFrame containing the offline off-policy evaluation scores for the given policy
        on the selected problem.
    """
    path_to_policy_output_directory = (
        path_to_project_root()
        / "data"
        / "for_policy_evaluation"
        / config.training.data.policy
        / policy_name
    )
    policy_output_df = pd.read_csv(
        path_to_policy_output_directory / f"{problem_id}.csv"
    )
    if "problem" in problem_id:
        # problem-level
        features = [
            "userID",
            "action",
            "problem",
            "reward",
            "problem_Q_value",
            "step_decision_Q_value",
            "example_Q_value",
        ]
    else:
        # step-level
        features = [
            "userID",
            "action",
            "problem",
            "reward",
            "problem_Q_value",
            "example_Q_value",
        ]
    df = policy_output_df[features]
    if "problem" in problem_id:
        df = df.rename(
            columns={
                "problem_Q_value": "ps",
                "step_decision_Q_value": "fwe",
                "example_Q_value": "we",
            }
        )
    else:
        df = df.rename(columns={"problem_Q_value": "ps", "example_Q_value": "we"})

    # action label: example 1, problem 0
    df["real_action"] = 1
    df["real_action"] = np.where(df["action"] == "problem", 0, df.real_action)
    if "problem" in problem_id:
        df["real_action"] = np.where(df["action"] == "example", 2, df.real_action)
        df = df[df["problem"] != "ex222"]
        df = df[df["problem"] != "ex144"]

    test = ImportanceSampling(df, 0.1, 0.9, policy_name)
    test.readData(problem_id=problem_id)

    static_info: Dict[str, str] = {
        "Policy Name": [policy_name.upper()],
        "Problem ID": [problem_id],
    }
    metrics: Dict[str, float] = {
        ope: [getattr(test, ope)()]
        for ope in ["IS", "WIS", "PDIS", "PHWIS", "DR", "WDR"]
    }
    # merge the two dictionaries
    static_info.update(metrics)

    return pd.DataFrame.from_dict(static_info)


def evaluate_all_policies(config: Config = None) -> None:
    """
    Evaluate the DQN policies using the offline off-policy evaluation (OPE) methods.

    Args:
        config: The configuration settings.

    Returns:
        None
    """
    if config is None:
        config = load_configuration()
    problems = ["problem"]
    problems.extend(list(config.training.problems))

    for algorithm_str in config.training.algorithms:
        ope_scores_dfs: List[pd.DataFrame] = []
        for problem_id in problems:
            if problem_id in config.training.skip.problems:
                continue  # skip the problem; it is not used for training
            if "problem" not in problem_id:
                problem_id += "(w)"
            print(
                f"{Fore.YELLOW}"
                f"Policy: {algorithm_str.upper()} | "
                f"Data: {config.training.data.policy.capitalize()}"
                f"{Style.RESET_ALL}"
            )
            with alive_bar(
                monitor=None,
                stats=None,
                title=f"ID: {problem_id}",
            ):
                temp_ope_scores_df: pd.DataFrame = (
                    evaluate_policy_with_importance_sampling(
                        policy_name=algorithm_str, problem_id=problem_id, config=config
                    )
                )
                ope_scores_dfs.append(temp_ope_scores_df)

        ope_scores_df = pd.concat(ope_scores_dfs)
        directory_path = (
            path_to_project_root()
            / "data"
            / "for_policy_evaluation"
            / config.training.data.policy
            / "for_analysis"
        )
        directory_path.mkdir(parents=True, exist_ok=True)
        ope_scores_df.to_csv(directory_path / f"{algorithm_str}.csv", index=False)
        print(
            f"{Fore.GREEN}"
            f"{algorithm_str.upper()} | Evaluation complete; results saved to "
            f"{directory_path / f'{algorithm_str}.csv'}"
            f"{Style.RESET_ALL}"
        )


if __name__ == "__main__":
    evaluate_all_policies()
