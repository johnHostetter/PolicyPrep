"""
This script takes all the semester data found in the 'with_delayed_rewards' folder and aggregates
it together for later inferring the immediate rewards.

The output is saved into a folder called 'for_inferring_rewards'.

This generated output produces a .csv file for problem-level and each problem that is available
for step-level decision-making.

Inferring immediate rewards is an essential process to our Reinforcement Learning policy
induction. This is because without inferring immediate rewards, all we are left with is a delayed
reward, which is the students' calculated Normalized Learning Gain (NLG). In essence, it makes it
harder to figure out during the sequence what was a good decision and what was a bad decision.

Inferring immediate rewards can be done in a few ways, such as Gaussian processes (older research
has done it this way) and something called an InferNet. InferNet is essentially a Long-Short Term
Memory (LSTM) trained to predict a students' NLG.

When training InferNet, first take the students' NLG and try to learn a LSTM that can predict the
NLG on the problem-level. Then, at each problem-level decision point, the LSTM can calculate a
predicted NLG.

This predicted NLG at the problem-level decision point is used as the target value for the end of a
step-level trajectory. As a result, after training a problem-level LSTM, for each problem,
the estimate problem-level NLG at each decision point is used to construct and predict the
step-level immediate rewards.
"""
import pandas as pd

from src.utils.reproducibility import load_configuration, path_to_project_root


def iterate_over_semester_data(subdirectory: str, problem_id: str) -> None:
    """
    Iterate over the different semesters of training data and generate the training data with action
    and reward columns.

    Returns:
        None
    """
    # load the configuration settings
    pd.options.mode.chained_assignment = None
    config = load_configuration("default_configuration.yaml")

    # iterate over the different semesters of training data

    semester_folder_path_generator = (
        path_to_project_root() / "data" / subdirectory
    ).glob("* - *")
    data_frames = []
    for semester_folder in semester_folder_path_generator:
        if not semester_folder.is_dir():
            print(f"Skipping {semester_folder.name} (not a directory)...")
            continue
        if " - " not in semester_folder.name:
            print(f"Skipping {semester_folder.name} (invalid directory name)...")
            continue

        # the name of the semester is the part of the folder name after the " - "
        # e.g. "10 - S21" -> "S21"; the part before the " - " is the folder ordering number
        (_, semester_name) = semester_folder.name.split(" - ")
        print(f"Processing data for the {semester_name} semester...")

        try:
            data_frame = pd.read_csv(semester_folder / f"{problem_id}.csv")
            data_frames.append(data_frame)

        except FileNotFoundError as file_not_found_error:
            print(
                f"Skipping {semester_folder.name} (no {file_not_found_error.filename} file "
                f"found for this semester)..."
            )
            continue

    concat_dataframes_and_save(data_frames, problem_id)


def concat_dataframes_and_save(data_frames: list, problem_id: str) -> None:
    """
    Args:
        data_frames: The list of data frames to concatenate and save to a .csv file.
        problem_id: The problem ID.

    Returns:
        None
    """
    if len(data_frames) > 0:
        data_frame = pd.concat(data_frames)
        print(f"Data for {problem_id} has shape {data_frame.shape}")

        if (
            "problem" in problem_id
        ):  # problem-level data is ready to infer immediate rewards
            subdirectory = "for_inferring_rewards"
        else:  # exercise data is not ready to have immediate rewards inferred
            subdirectory = "for_propagating_rewards"
        # make the output directories for the training data
        output_directory = path_to_project_root() / "data" / subdirectory
        output_directory.mkdir(parents=True, exist_ok=True)

        # save the data to a .csv file
        data_frame.to_csv(output_directory / f"{problem_id}.csv", index=False)


def aggregate_data_for_inferring_rewards():
    """
    Aggregate the data for inferring rewards for all problems.

    Returns:
        None
    """
    # load the configuration settings
    config = load_configuration("default_configuration.yaml")

    problems = [f"{problem_id}(w)" for problem_id in config.training.problems]
    problems.insert(0, "problem")

    # iterate over the different problems
    for problem_id in problems:
        if problem_id == "problem":
            print(f"Aggregating data for problem-level...")
        else:
            print(f"Aggregating data for problem {problem_id}...")
        iterate_over_semester_data("with_delayed_rewards", problem_id)


if __name__ == "__main__":
    aggregate_data_for_inferring_rewards()
