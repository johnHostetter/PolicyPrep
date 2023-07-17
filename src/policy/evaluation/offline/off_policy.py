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
import numpy as np
import pandas as pd

from YACS.yacs import Config
from src.utils.reproducibility import path_to_project_root, load_configuration
from src.utils.importance_sampling import ImportanceSampling


def evaluate_policy_with_importance_sampling(policy_name: str, problem_id: str):
    """
    Evaluate the given policy on the selected problem ID; if problem_id is "problem",
    then evaluate the given policy on the problem-level scenario.

    Args:
        policy_name:
        problem_id:

    Returns:

    """
    print("Loading data...")
    path_to_policy_output_directory = (
            path_to_project_root() / "data" / "for_policy_evaluation" / policy_name
    )
    policy_output_df = pd.read_csv(
        path_to_policy_output_directory / f"{problem_id}.csv"
    )
    print("Finished loading data.")

    if "problem" in problem_id:
        # problem-level
        features = [
            "userID",
            "action",
            "problem",
            "inferred_reward",
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
            "inferred_reward",
            "problem_Q_value",
            "example_Q_value",
        ]

    df = policy_output_df[features]
    if "problem" in problem_id:
        df = df.rename(
            columns={"problem_Q_value": "ps", "step_decision_Q_value": "fwe", "example_Q_value": "we"}
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

    for ope in ["IS", "WIS", "PDIS", "PHWIS", "DR", "WDR"]:
        value = getattr(test, ope)()

        print("{},{},{}".format(policy_name, ope, value))


def evaluate_dqn_policies(config: Config = None) -> None:
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
    for problem_id in problems:
        if problem_id in config.training.skip.problems:
            continue  # skip the problem; it is not used for training
        if "problem" not in problem_id:
            problem_id += "(w)"
        print(f"Evaluating the policy on {problem_id}...")
        evaluate_policy_with_importance_sampling(
            policy_name="dqn", problem_id=problem_id
        )


if __name__ == "__main__":
    evaluate_dqn_policies()
