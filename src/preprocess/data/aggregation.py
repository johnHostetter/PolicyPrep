"""
This script takes all the semester data found in the 'with_delayed_rewards'
folder and aggregates it together for later inferring the immediate rewards.

The output is saved into a folder called 'for_inferring_rewards'.

This generated output produces a .csv file for problem-level and each
problem that is available for step-level decision-making.

Inferring immediate rewards is an essential process to our Reinforcement
Learning policy induction. This is because without inferring immediate
rewards, all we are left with is a delayed reward, which is the students'
calculated Normalized Learning Gain (NLG). In essence, it makes it harder
to figure out during the sequence what was a good decision and what
was a bad decision.

Inferring immediate rewards can be done in a few ways,
such as Gaussian processes (older research has done it this way)
and something called an InferNet. InferNet is essentially a
Long-Short Term Memory (LSTM) trained to predict a students' NLG.

When training InferNet, first take the students' NLG and try to learn
a LSTM that can predict the NLG on the problem-level. Then, at each
problem-level decision point, the LSTM can calculate a predicted NLG.

This predicted NLG at the problem-level decision point is used as the
target value for the end of a step-level trajectory. As a result,
after training a problem-level LSTM, for each problem, the estimate
problem-level NLG at each decision point is used to construct and predict
the step-level immediate rewards.
"""

import os
import pandas as pd

from utils.reproducibility import load_configuration, path_to_project_root


def aggregate_data_for_inferring_rewards() -> None:
    """
    This script takes all the semester data found in the 'with_delayed_rewards'
    folder and aggregates it together for later inferring the immediate rewards.

    Returns:
        None
    """
    config = load_configuration()
    problems = ['problem']
    problems.extend(config.training.problems)

    semesters = os.listdir(path_to_project_root() / "data" / 'with_delayed_rewards')
    semesters = [semester for semester in semesters if '.zip' not in semester]

    for problem in problems:
        semester, dataframes = None, []  # a list of Pandas DataFrames to be concatenated together
        for semester in semesters:
            if '.zip' in semester:
                continue  # ignore compressed files
            try:
                if problem == 'problem':
                    df = pd.read_csv('with_delayed_rewards/{}/{}-action+reward.csv'.format(semester, problem))
                else:
                    df = pd.read_csv('with_delayed_rewards/{}/{}(w)-action+reward.csv'.format(semester, problem))
                dataframes.append(df)
            except FileNotFoundError:
                print('File not found for the semester {} regarding {}.'.format(semester, problem))
                continue
        if len(dataframes) > 0:
            df = pd.concat(dataframes)
            print('Data for {} has shape {}'.format(problem, df.shape))

            # make the output directories for the training data
            output_directory = (
                path_to_project_root() / "data" / "for_inferring_rewards"
            )
            output_directory.mkdir(parents=True, exist_ok=True)

            # save the combined data to be used for inferring immediate rewards (e.g., InferNet)
            df.to_csv('for_inferring_rewards/{}-action+reward.csv'.format(problem), index=False)
