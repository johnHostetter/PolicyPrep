"""
This script loads a trained InferNet model and uses it to infer the rewards for a hypothetical
action. This is helpful when trying to understand the impact of a particular action. For example,
if you want to know what the reward would be if you changed the action from "elicit" to "tell",
you can use this script to find out.
"""
from typing import Union

import pandas as pd

from src.preprocess.infernet.train import infernet_setup, get_features_and_actions
from src.preprocess.infernet.common import (
    load_infernet_model,
    create_buffer,
    normalize_data,
    infer_and_save_rewards,
)
from src.preprocess.data.selection import get_most_recent_file
from src.utils.reproducibility import load_configuration, path_to_project_root


def use_infer_net(problem_id: str, hypothetical_action: str) -> pd.DataFrame:
    """
    Use the InferNet model to infer the rewards for a hypothetical action. Helpful to use when
    trying to understand the impact of a particular action.

    Args:
        problem_id: The name of the problem or exercise, or "problem" if the data is problem-level.
        hypothetical_action: The action to use when inferring the rewards.

    Returns:
        A DataFrame with the inferred rewards for the hypothetical action.
    """
    # load the configuration file
    config = load_configuration()

    is_problem_level, max_len, original_data, user_ids = infernet_setup(problem_id)

    # select the features and actions depending on if the data is problem-level or step-level
    state_features, possible_actions = get_features_and_actions(
        config, is_problem_level
    )

    # normalize the data
    normalized_data = normalize_data(
        original_data, problem_id, columns_to_normalize=state_features
    )

    # modify the data to use the hypothetical action
    if hypothetical_action in possible_actions:
        normalized_data["action"] = hypothetical_action
    elif hypothetical_action == "correct":
        # use the correct action, no need to modify the data with a hypothetical action
        pass  # do nothing
    else:
        raise ValueError(
            f"{hypothetical_action} is not a valid action for {problem_id}."
        )

    # create the buffer to train InferNet from
    print("Creating buffer...")
    infer_buffer = create_buffer(
        normalized_data,
        user_ids,
        config,
        is_problem_level=is_problem_level,
        max_len=max_len,  # max_len is the max episode length; not required for problem-level data
    )

    # load the most recent model
    print("Loading model...")
    path_to_model = get_most_recent_file(
        path_to_folder="models", problem_id=problem_id, file_type="h5"
    )
    model = load_infernet_model(path_to_model)

    # infer the rewards
    print(
        f"Inferring rewards for the hypothetical action of using {hypothetical_action} in"
        f" {problem_id}..."
    )
    return infer_and_save_rewards(
        problem_id,
        0,
        infer_buffer,
        max_len,
        model,
        state_features,
        num_state_and_actions=len(state_features) + len(possible_actions),
        is_problem_level=is_problem_level,
        inferred_reward_column_name=f"inferred_reward_{hypothetical_action}",
        save_inferred_rewards=False,
    )


if __name__ == "__main__":
    problem_id = "ex132(w)"
    config = load_configuration()
    _, possible_actions = get_features_and_actions(
        config, is_problem_level="problem" in problem_id
    )
    results_df: Union[None, pd.DataFrame] = None
    possible_actions = list(possible_actions)
    possible_actions.insert(0, "correct")  # start by using the correct action
    for action in possible_actions:
        print(
            f"Using InferNet to infer rewards for the hypothetical action of using {action} in "
            f"{problem_id}..."
        )
        new_df = use_infer_net(problem_id=problem_id, hypothetical_action=action)
        print(
            f"Finished inferring rewards for the hypothetical action of using {action} in "
            f"{problem_id}."
        )
        if results_df is None:
            results_df = new_df
        else:
            columns_to_use = new_df.columns.difference(results_df.columns)
            results_df = pd.merge(
                results_df,
                new_df[columns_to_use],
                left_index=True,
                right_index=True,
                how="outer",
            )
    print(
        f"Saving results to data/each_action_with_inferred_rewards/{problem_id}.csv..."
    )
    output_directory = (
        path_to_project_root() / "data" / "each_action_with_inferred_rewards"
    )
    output_directory.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_directory / f"{problem_id}.csv", index=False)
    print(
        f"Finished saving results to data/each_action_with_inferred_rewards/{problem_id}.csv."
    )
