"""
This module implements the pipeline for the project. The pipeline is as follows:
    (1) Load the configuration file.
    (2) Download the semester data from the Google Drive folder.
    (3) Preprocess the data to make it compatible with InferNet.
    (4) Aggregate the data into a single file; for example, one file for problem-level data
    and one file for each exercises' step-level data.
    (5) Train the InferNet model for the problem-level.
    (6) Propagate the problem-level rewards to the step-level.
    (7) Train the InferNet model for each exercise (step-level) data file that was created in
    step (4) above (except for the problem-level data file) using the InferNet model trained
    in step (5) above to infer the immediate rewards for each exercise (step-level).
    (8) Select the most recent data file (with inferred immediate rewards) produced as a result
    of training InferNet, and store it in the data subdirectory called "for_policy_induction".
    (9) Train the policy induction model using the data files selected in step (8) above.

This module will run the entire pipeline. During the execution of the pipeline, the data will be
downloaded from the Google Drive folder into a subdirectory of the data folder called "raw". The
data will then be preprocessed to make it compatible with InferNet. The data will then be
aggregated into a single file, and saved to a subdirectory of the data folder called
"for_inferring_rewards" or "for_propagating_rewards" depending on whether the data is
problem-level or step-level data. The problem-level data will be used to train the InferNet model
for the problem-level. The problem-level rewards will then be propagated to the step-level. The
step-level data will then be used to train the InferNet model for each exercise (step-level) data
file that was created in step (4) above (except for the problem-level data file) using the
InferNet model trained in step (5) above to infer the immediate rewards for each exercise (
step-level).
TODO: During all the above steps, the data may be normalized and/or standardized. The data will
TODO: be saved to a subdirectory of the data folder called "normalized" or "standardized"
TODO: depending on whether the data was normalized or standardized.

Warning: If you run this script within an IDE, it will run the pipeline, which will take a long time to
complete (days or weeks). If you want to run the pipeline, run the following command from the
project root directory:

    python src/pipeline.py

Further, if you want to run the pipeline, you should run it from the command line, not from an IDE. If
you run the pipeline from an IDE, the IDE may crash due to memory issues. If you run the pipeline from
the command line, the pipeline will run in the background, and you can continue to use your computer
while the pipeline is running.

If you want to run a single step of the pipeline, run the corresponding module as a script.
For example, if you want to run step (3) above, run the following command from the project root
directory:
    python -m src.preprocess.data.to_infernet_format

If you want to run a single step of the pipeline for a specific exercise, run the corresponding
module as a script. For example, if you want to run step (7) above for the exercise "problem",
run the following command from the project root directory:
    python -m src.preprocess.infernet.train --problem_id problem
"""
import multiprocessing as mp

from src.preprocess.data.lookup import lookup_semester_grades_and_append_if_missing
from src.preprocess.infernet.train import train_infer_net
from src.preprocess.data.download import download_semester_data
from src.preprocess.data.aggregation import aggregate_data_for_inferring_rewards
from src.preprocess.data.selection import select_training_data_for_policy_induction
from src.preprocess.data.to_infernet_format import (
    iterate_over_semester_data,
    convert_data_format,
)
from src.preprocess.infernet.problem_to_step import (
    propagate_problem_level_rewards_to_step_level,
)
from src.utils.reproducibility import (
    load_configuration,
)  # order matters, must be imported last

if __name__ == "__main__":
    # load the configuration file
    config = load_configuration()

    # download the data from the Google Drive folder into a subdirectory of the
    # data folder called "raw"
    download_semester_data()

    iterate_over_semester_data(
        subdirectory="raw",
        function_to_perform=lookup_semester_grades_and_append_if_missing,
    )

    # preprocess the data to make it compatible with InferNet
    iterate_over_semester_data(
        subdirectory="raw", function_to_perform=convert_data_format
    )

    # aggregate the data into a single file
    aggregate_data_for_inferring_rewards()

    # train the InferNet model for the problem level data
    train_infer_net(problem_id="problem")

    # propagate problem-level rewards to step-level rewards
    propagate_problem_level_rewards_to_step_level()

    # train an InferNet model for each exercise
    num_workers = mp.cpu_count() - 1
    with mp.Pool(processes=num_workers) as pool:
        for problem_id in config.training.problems:
            if problem_id not in config.training.skip.problems:
                pool.apply_async(train_infer_net, args=(f"{problem_id}(w)",))
        pool.join()

    print("All processes finished for training step-level InferNet models.")

    # select the training data to be used for policy induction via offline reinforcement learning
    select_training_data_for_policy_induction()
