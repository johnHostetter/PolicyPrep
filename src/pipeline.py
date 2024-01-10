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

    The --run_specific flag is not set by default. If it is not set,  all the code (i.e., steps) after the
    specified step will be executed. If you only want to execute the code for the specified step,
    set the --run_specific flag. For example, if you want to run step (3) above, and only that step,
    run the following command from the project root directory:

        python -m src/preprocess/data/to_infernet_format --step 3 --run_specific

    This will execute the code for step (3) above only. If you do not set the --run_specific flag,
    the code for all steps after the specified step will be executed. For example, if you specify
    step 3 and do not set the --run_specific flag, the code for steps 3, 4, 5, 6, 7, 8, and 9 will be
    executed. If you specify step 5 and set the --run_specific flag, the code for only step 5
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
import shutil
from pathlib import Path

from colorama import (
    init as colorama_init,
)  # for cross-platform colored text in the terminal
from colorama import Fore, Style  # for cross-platform colored text in the terminal

from src.policy.inference.with_d3rlpy_dqn import calculate_d3rlpy_algo_q_values
from src.policy.evaluation.offline.off_policy import evaluate_all_policies
from src.policy.induction.d3rlpy.dqn import induce_policies_with_d3rlpy
from src.preprocess.data.lookup import lookup_semester_grades_and_append_if_missing
from src.preprocess.data.make_consistent import append_year_and_semester_to_user_id
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
from src.utilities.reproducibility import (
    parse_keyword_arguments,
    load_configuration,
    path_to_project_root,
)  # order matters, must be imported last


if __name__ == "__main__":
    colorama_init()  # initialize colorama for cross-platform colored text in the terminal

    # parse the command line arguments
    args = parse_keyword_arguments()
    print(args)

    # load the configuration file
    config = load_configuration(args.config_file_path)

    # download the data from the Google Drive folder into a subdirectory of the
    # data folder called "raw"
    if args.step == 1:
        print(
            f"{Fore.CYAN}"
            f"(1): Downloading the semester data from the Google Drive folder..."
            f"{Style.RESET_ALL}"
        )
        download_semester_data()

    # begin by moving all the files from the raw directory to the clean directory
    # (some files may be overwritten later, but this is okay and easier)
    if args.step == 2 or (not args.run_specific and args.step <= 2):
        print(
            f"{Fore.CYAN}"
            f"(2): Copying all the files from the raw directory to the clean directory..."
            f"{Style.RESET_ALL}"
        )
        path_to_raw_directory: Path = path_to_project_root() / "data" / "raw"
        path_to_clean_directory: Path = path_to_project_root() / "data" / "clean"
        # delete the clean directory if it exists
        shutil.rmtree(path_to_clean_directory, ignore_errors=True)
        # copy the raw directory to the clean directory (this will overwrite any files that
        # already exist in the clean directory)
        shutil.copytree(path_to_raw_directory, path_to_clean_directory)

    # add the year and semester to the user IDs if they are not already there
    if args.step == 3 or (not args.run_specific and args.step <= 3):
        print(
            f"{Fore.CYAN}"
            f"(3): Adding the year and semester to the user IDs if they are not already there..."
            f"{Style.RESET_ALL}"
        )
        iterate_over_semester_data(
            subdirectory="raw",
            function_to_perform=append_year_and_semester_to_user_id,
            config=config,
        )

    # find the semester data files and lookup the grades for each student and
    # append them to the data files if they are missing (this modifies files in data/raw)
    if args.step == 4 or (not args.run_specific and args.step <= 4):
        print(
            f"{Fore.CYAN}"
            "(4): Looking up the grades for each student and appending them to the "
            "data files if they are missing..."
            f"{Style.RESET_ALL}"
        )
        iterate_over_semester_data(
            subdirectory="clean",
            function_to_perform=lookup_semester_grades_and_append_if_missing,
            config=config,
        )

    # # clean the data by removing the offset used for wording problems and removing
    # # any users that have NaN grades, as well as adding the year and semester to the
    # # user IDs if they are not already there
    # if args.step == 4 or (not args.run_specific and args.step <= 4):
    #     print(
    #         "(3): Cleaning the data by removing the offset used for wording problems and "
    #         "removing any users that have NaN grades..."
    #     )
    #     iterate_over_semester_data(
    #         subdirectory="raw", function_to_perform=clean_semester_data, config=config
    #     )

    # preprocess the data to make it compatible with InferNet
    if args.step == 5 or (not args.run_specific and args.step <= 5):
        print(
            f"{Fore.CYAN}"
            "(5): Preprocessing the data to make it compatible with InferNet..."
            f"{Style.RESET_ALL}"
        )
        iterate_over_semester_data(
            subdirectory="clean", function_to_perform=convert_data_format, config=config
        )

    # aggregate the data into a single file
    if args.step == 6 or (not args.run_specific and args.step <= 6):
        print(
            f"{Fore.CYAN}"
            "(6): Aggregating the data into a single file..."
            f"{Style.RESET_ALL}"
        )
        aggregate_data_for_inferring_rewards()

    # train the InferNet model for the problem level data
    if args.step == 7 or (not args.run_specific and args.step <= 7):
        print(
            f"{Fore.CYAN}"
            "(7): Training the InferNet model for the problem level data..."
            f"{Style.RESET_ALL}"
        )
        train_infer_net(problem_id="problem")

    # propagate problem-level rewards to step-level rewards
    if args.step == 8 or (not args.run_specific and args.step <= 8):
        print(
            f"{Fore.CYAN}"
            "(8): Propagating problem-level rewards to step-level rewards..."
            f"{Style.RESET_ALL}"
        )
        propagate_problem_level_rewards_to_step_level(num_workers=args.num_workers)

    # train an InferNet model for each exercise
    if args.step == 9 or (not args.run_specific and args.step <= 9):
        print(
            f"{Fore.CYAN}"
            "(9): Training the InferNet model for each exercise (step-level) data file..."
            f"{Style.RESET_ALL}"
        )

        train_step_level_models(args, config)

    # select the training data to be used for policy induction via offline reinforcement learning
    if args.step == 10 or (not args.run_specific and args.step <= 10):
        print(
            f"{Fore.CYAN}"
            "(10) Selecting the training data to be used for policy induction "
            "via offline reinforcement learning..."
            f"{Style.RESET_ALL}"
        )
        select_training_data_for_policy_induction(num_workers=args.num_workers)

    # induce policies via selected offline reinforcement learning algorithms & training data
    if args.step == 11 or (not args.run_specific and args.step <= 11):
        print(
            f"{Fore.CYAN}"
            "(11) Training the policy induction model using the selected training data..."
            f"{Style.RESET_ALL}"
        )
        induce_policies_with_d3rlpy()

        # calculate the Q-values for the induced policies
    if args.step == 12 or (not args.run_specific and args.step <= 12):
        print(
            f"{Fore.CYAN}"
            "(12) Calculating the Q-values for the induced policies..."
            f"{Style.RESET_ALL}"
        )
        calculate_d3rlpy_algo_q_values()

        # evaluate the induced policies using their respective calculated Q-values
    if args.step == 13 or (not args.run_specific and args.step <= 13):
        print(
            f"{Fore.CYAN}"
            "(13) Evaluating the policy induction model using the selected training data..."
            f"{Style.RESET_ALL}"
        )
        evaluate_all_policies()
