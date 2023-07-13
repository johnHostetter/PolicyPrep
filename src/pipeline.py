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

*** IMPORTANT ***

How to run the pipeline:

    Warning: If you run this script within an IDE, it will run the pipeline, which will take a long
    time to complete (days or weeks). If you want to run the pipeline, run the following command
    from the project root directory:

        python src/pipeline.py

How to run the pipeline with an IDE or use a different configuration file:

    You should only run the pipeline from an IDE if you want to run a single step of the
    pipeline, or if you have edited the default configuration file and want to run the pipeline
    with the edited configuration file. If you want to run a single step of the pipeline, see the
    instructions below. If you have edited the default configuration file and want to run the
    pipeline with the edited configuration file, run the following command from the project root
    directory:

        python src/pipeline.py --config_file_path <path_to_config_file>

    For example, if you have edited the default configuration file and saved it to the following
    path:

        src/config/edited_configuration.yaml

    run the following command from the project root directory:

        python src/pipeline.py --config_file_path src/config/edited_configuration.yaml

    Such edits may include a reduced training time spent in InferNet. For example, if you want to
    reduce the training time spent in InferNet from 1000 epochs to 100 epochs, you can edit the
    default configuration file to set the number of epochs to 100.

    Additional edits may include disabling multiprocessing. For example, if you want to disable
    multiprocessing, you can specify the number of workers to 1 by the --num_workers flag (see
    "About the multiprocessing").

How to run a step of the pipeline & all steps after that step:

    Sometimes, we may have completed some steps of the pipeline, and we may want to run the
    remaining steps of the pipeline. For example, we may have already downloaded the data from the
    Google Drive folder, and we may want to run the remaining steps of the pipeline. In this case,
    we can run the pipeline with the --step flag followed by the step number. For example, if we
    want to run the remaining steps of the pipeline, we can run the following command from the
    project root directory:

        python src/pipeline.py --step 2

    This will execute the code for step (2) above and all steps after that step. For example, if
    we specify step 2, the code for steps 2, 3, 4, 5, 6, 7, 8, and 9 will be executed. If we
    specify step 5, the code for steps 5, 6, 7, 8, and 9 will be executed. If we specify step 9,
    the code for step 9 will be executed. The --step flag is set to 1 by default. If you don't
    specify the --step flag, the code for step 1 will be executed. If you specify the --step
    flag, the code for the step you specified and all steps after that step will be executed.

    In some cases, you can run a single step of the pipeline by running the corresponding module
    as a script. For example, if you want to run step (3) above, run the following command from
    the project root directory:

        python -m src/preprocess/data/to_infernet_format

    Alternatively, you can also specify the step you want to run by adding the --step flag
    followed by the step number through the pipeline.py script. For example, if you want to run
    step (3) above, run the following command from the project root directory:

        python -m src/preprocess/data/to_infernet_format --step 3

    This will execute the code for step (3) above and all steps after that step. For example, if you
    specify step 3, the code for steps 3, 4, 5, 6, 7, 8, and 9 will be executed. If you specify
    step 5, the code for steps 5, 6, 7, 8, and 9 will be executed. If you specify step 9, the code
    for step 9 will b executed.

How to run a single step of the pipeline:

    The --run_all flag is set to True by default, where all the code (i.e., steps) after the
    specified step will be executed. If you only want to execute the code for the specified step,
    set the --run_all flag to False. For example, if you want to run step (3) above, run the
    following command from the project root directory:

        python -m src/preprocess/data/to_infernet_format --step 3 --run_all False

    This will execute the code for step (3) above only. If you set the --run_all flag to True,
    the code for all steps after the specified step will be executed. For example, if you specify
    step 3 and set the --run_all flag to True, the code for steps 3, 4, 5, 6, 7, 8, and 9 will be
    executed. If you specify step 5 and set the --run_all flag to False, the code for only step 5
    will be executed.

About the multiprocessing:

    Optionally, you can specify the number of workers to use for multiprocessing by adding the
    --num_workers flag followed by the number of workers to use. For example, if you want to use 4
    workers, run the following command from the project root directory:

        python src/pipeline.py --num_workers 4

    Note that the number of workers should be less than or equal to the number of cores on your
    computer. If you don't specify the number of workers, the number of workers will be set to the
    number of cores on your computer. If you specify a number of workers that is greater than the
    number of cores on your computer, the number of workers will be set to the number of cores on
    your computer. If you specify a number of workers that is less than or equal to the number of
    cores on your computer, the number of workers will be set to the number of workers you
    specified.

    Note that you can't use the --num_workers flag in an IDE. You should only use this flag if
    you run the pipeline from the command line. If you run the pipeline from an IDE, the IDE may
    crash due to memory issues. If you run the pipeline from the command line, the pipeline will
    run in the background, and you can continue to use your computer while the pipeline is running.

How to run a single step of the pipeline that involves multiprocessing:

    Some steps involve multiprocessing. If you want to run a single step of the pipeline that
    involves multiprocessing, run the corresponding module as a script, and specify the number of
    workers to use for multiprocessing by adding the --num_workers flag followed by the number of
    workers to use. For example, if you want to run step (4) above, run the following command from
    the project root directory:

        python -m src/preprocess/data/aggregation --num_workers 4

    Note that the number of workers should be less than or equal to the number of cores on your
    computer. If you don't specify the number of workers, the number of workers will be set to the
    number of cores on your computer. If you specify a number of workers that is greater than the
    number of cores on your computer, the number of workers will be set to the number of cores on
    your computer. If you specify a number of workers that is less than or equal to the number of
    cores on your computer, the number of workers will be set to the number of workers you
    specified.

    You may also specify the number of workers to use throughout the entire execution of the
    pipeline by adding the --num_workers flag followed by the number of workers to use through the
    pipeline.py script. For example, if you want to run the pipeline with 4 workers, run the
    following command from the project root directory:

    python -m src/pipeline --num_workers 4

How to run a single step of the pipeline for a specific exercise:

    If you want to run a single step of the pipeline for a specific exercise, run the corresponding
    module as a script. For example, if you want to run step (7) above for the exercise "problem",
    run the following command from the project root directory (TODO: test or implement this):

        python -m src/preprocess/infernet/train --problem_id problem
"""
import argparse
import multiprocessing as mp

from src.policy.induction.d3rlpy.dqn import induce_dqn_policies
from src.preprocess.data.lookup import lookup_semester_grades_and_append_if_missing
from src.preprocess.infernet.train import (
    train_infer_net,
    train_step_level_models,
)
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


def parse_keyword_args() -> argparse.Namespace:
    """
    Parse the keyword arguments passed to the script.

    Returns:
        The keyword arguments passed to the script.
    """
    parser = argparse.ArgumentParser(
        description="Run the pipeline for the project. The pipeline is as follows: "
        "(1) Load the configuration file. "
        "(2) Download the semester data from the Google Drive folder. "
        "(3) Preprocess the data to make it compatible with InferNet. "
        "(4) Aggregate the data into a single file; for example, one file for problem-level data "
        "and one file for each exercises' step-level data. "
        "(5) Train the InferNet model for the problem-level. "
        "(6) Propagate the problem-level rewards to the step-level. "
        "(7) Train the InferNet model for each exercise (step-level) data file that was created in "
        "step (4) above (except for the problem-level data file) using the InferNet model trained "
        "in step (5) above to infer the immediate rewards for each exercise (step-level). "
        "(8) Select the most recent data file (with inferred immediate rewards) "
        "produced as a result "
        "of training InferNet, and store it in the data subdirectory called "
        '"for_policy_induction". '
        "(9) Train the policy induction model using the data files selected in step (8) above.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--step",
        type=int,
        default=1,
        help="The step of the pipeline to run. "
        "1: download the semester data from the Google Drive folder. "
        "2: preprocess the data to make it compatible with InferNet. "
        "3: aggregate the data into a single file; for example, one file for problem-level data "
        "and one file for each exercises' step-level data. "
        "4: train the InferNet model for the problem-level. "
        "5: propagate the problem-level rewards to the step-level. "
        "6: train the InferNet model for each exercise (step-level) data file that was created in "
        "step (4) above (except for the problem-level data file) using the InferNet model trained "
        "in step (5) above to infer the immediate rewards for each exercise (step-level). "
        "7: select the most recent data file (with inferred immediate rewards) "
        "produced as a result "
        "of training InferNet, and store it in the data subdirectory called "
        '"for_policy_induction". '
        "8: train the policy induction model using the data files selected in step (7) above.",
    )
    parser.add_argument(
        "--run_all",
        type=bool,
        default=True,
        help="If True, continue with steps after the step that is specified "
        "by the --step argument. "
        "If False, do not continue with steps after the step that is specified "
        "by the --step argument.",
    )
    parser.add_argument(
        "--problem_id",
        type=str,
        default="problem",
        help="The problem ID for which to run the pipeline. "
        "If the step is 1, 2, 3, 4, 5, or 7, this argument is ignored. "
        "If the step is 6, this argument is required. "
        "If the step is 8, this argument is optional. "
        "If the step is 9, this argument is required.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=mp.cpu_count() - 1,
        help="The number of workers (i.e., processes) to use for multiprocessing. "
        "If the step is 1, 2, 3, 4, 5, or 7, this argument is ignored. "
        "If the step is 6, this argument is optional. "
        "If the step is 8, this argument is optional. "
        "If the step is 9, this argument is required.",  # TODO: make this optional?
    )

    parser.add_argument(
        "--config_file_path",
        type=str,
        default="default_configuration.yaml",
        help="The path to the configuration file.",
    )

    arguments = parser.parse_args()
    arguments.num_workers = min(arguments.num_workers, mp.cpu_count() - 1)

    return arguments


if __name__ == "__main__":
    # parse the command line arguments
    args = parse_keyword_args()

    print(args)

    # load the configuration file
    config = load_configuration(args.config_file_path)

    # download the data from the Google Drive folder into a subdirectory of the
    # data folder called "raw"
    if args.step == 1:
        print("(1): Downloading the semester data from the Google Drive folder...")
        download_semester_data()

    # find the semester data files and lookup the grades for each student and
    # append them to the data files if they are missing
    if args.step == 2 or (args.run_all and args.step <= 2):
        print(
            "(2): Looking up the grades for each student and appending them to the "
            "data files if they are missing..."
        )
        iterate_over_semester_data(
            subdirectory="raw",
            function_to_perform=lookup_semester_grades_and_append_if_missing,
        )

    # preprocess the data to make it compatible with InferNet
    if args.step == 3 or (args.run_all and args.step <= 3):
        print("(3): Preprocessing the data to make it compatible with InferNet...")
        iterate_over_semester_data(
            subdirectory="raw", function_to_perform=convert_data_format
        )

    # aggregate the data into a single file
    if args.step == 4 or (args.run_all and args.step <= 4):
        print("(4): Aggregating the data into a single file...")
        aggregate_data_for_inferring_rewards()

    # train the InferNet model for the problem level data
    if args.step == 5 or (args.run_all and args.step <= 5):
        print("(5): Training the InferNet model for the problem level data...")
        train_infer_net(problem_id="problem")

    # propagate problem-level rewards to step-level rewards
    if args.step == 6 or (args.run_all and args.step <= 6):
        print("(6): Propagating problem-level rewards to step-level rewards...")
        propagate_problem_level_rewards_to_step_level(num_workers=args.num_workers)

    # train an InferNet model for each exercise
    if args.step == 7 or (args.run_all and args.step <= 7):
        print(
            "(7): Training the InferNet model for each exercise (step-level) data file..."
        )

        train_step_level_models(args, config)

    # select the training data to be used for policy induction via offline reinforcement learning
    if args.step == 8 or (args.run_all and args.step <= 8):
        print(
            "(8) Selecting the training data to be used for policy induction "
            "via offline reinforcement learning..."
        )
        select_training_data_for_policy_induction(num_workers=args.num_workers)

    # induce policies via selected offline reinforcement learning algorithms & training data
    if args.step == 9 or (args.run_all and args.step <= 9):
        print(
            "(9) Training the policy induction model using the selected training data..."
        )
        induce_dqn_policies()
