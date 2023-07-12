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
import pandas as pd
import numpy as np

from src.utils.reproducibility import path_to_project_root
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
    path_to_policy_output_directory = path_to_project_root() / "output" / "policy" / policy_name
    policy_output_df = pd.read_csv(path_to_policy_output_directory / f"{problem_id}.csv")
    print("Finished loading data.")

    if "problem" in problem_id:
        # problem-level
        features = [
            "userID",
            "action",
            "problem",
            "inferred_reward",
            "ps_Q_value",
            "fwe_Q_value",
            "we_Q_value",
        ]
    else:
        # step-level
        features = [
            "userID",
            "action",
            "problem",
            "inferred_reward",
            "elicit_Q_value",
            "tell_Q_value",
        ]

    df = policy_output_df[features]
    if "problem" in problem_id:
        df = df.rename(
            columns={"ps_Q_value": "ps", "fwe_Q_value": "fwe", "we_Q_value": "we"}
        )
    else:
        df = df.rename(columns={"elicit_Q_value": "ps", "tell_Q_value": "we"})

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


if __name__ == "__main__":
    evaluate_policy_with_importance_sampling(policy_name="dqn", problem_id="ex132")
