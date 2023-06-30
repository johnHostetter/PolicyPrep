import os
import pandas as pd

from pathlib import Path
from constant import PROBLEM_FEATURES, STEP_FEATURES

pd.options.mode.chained_assignment = None

basic_columns = [
    'userID', 'problem', 'decisionPoint',
]


# basic_columns = [
#     'recordID', 'answerID', 'time', 'userID', 'problem',
#     'decisionID', 'decisionPoint', 'decisionOrdering',
#     'substepMode', 'KC', 'session', 'substepOrdering'
# ]

def make_year_and_semester_int(semester):
    if semester[0] == 'S':
        semester_int = 1000
    elif semester[0] == 'F':
        semester_int = 3000
    year_int = int(semester[1:]) * 10000
    return year_int, semester_int


def minimum_id(semester):
    year_int, semester_int = make_year_and_semester_int(semester)
    return year_int + semester_int + 100


semesters = os.listdir('data/studies')
semesters = [semester for semester in semesters if '.zip' not in semester]
for semester in semesters:
    Path('data/with_delayed_rewards/{}'.format(semester)).mkdir(parents=True, exist_ok=True)

    subSteps = pd.read_csv('data/studies/{}/substep_info.csv'.format(semester), header=0)

    usersNLG = pd.read_csv('data/weighted_F15_F22.csv', header=0)

    userCount = len(subSteps['userID'].unique())
    problems = subSteps['problem'].unique()
    problems = [
        p for p in problems if (p[-1] != 'w' and 'ex222' not in p and 'ex144' not in p)
    ]  # Exclude all wording problems and ex222 and ex144 who won't be trained for step-level anyway

    # Problem-Level Training Data with action and reward columns
    prob_features = pd.read_csv('data/studies/{}/features_all.csv'.format(semester),
                                header=0, usecols=basic_columns + PROBLEM_FEATURES)

    prob_features = prob_features[prob_features['userID'] > minimum_id(semester)]

    problemLevelFeatureFrame = prob_features[prob_features['decisionPoint'].isin(['probStart', 'probEnd'])]
    problemLevelFeatureFrame.drop(problemLevelFeatureFrame[(problemLevelFeatureFrame['decisionPoint'] == 'probEnd') & (
        ~problemLevelFeatureFrame['problem'].isin(['ex252', 'ex252w']))].index, inplace=True)
    problemLevelFeatureFrame["action"] = ""
    problemLevelFeatureFrame["reward"] = ""
    actionColumnLocation = problemLevelFeatureFrame.columns.get_loc('action')
    rewardColumnLocation = problemLevelFeatureFrame.columns.get_loc('reward')

    for i in range(len(problemLevelFeatureFrame)):
        userID = problemLevelFeatureFrame.iloc[i]['userID']
        decisionPoint = problemLevelFeatureFrame.iloc[i]['decisionPoint']
        if decisionPoint == 'probStart':
            problem = problemLevelFeatureFrame.iloc[i]['problem']
            uniqueActions = subSteps[(subSteps['userID'] == userID) & (subSteps['problem'] == problem)][
                'substepMode'].unique()
            if len(uniqueActions) == 2:
                problemLevelFeatureFrame.iat[i, actionColumnLocation] = 'step_decision'
            else:
                try:
                    problemLevelFeatureFrame.iat[i, actionColumnLocation] = uniqueActions[0]
                except IndexError:
                    problemLevelFeatureFrame.iat[i, actionColumnLocation] = uniqueActions
        else:
            if len(usersNLG[usersNLG['userID'] == userID]['nlg'].unique()) == 0:
                continue  # the result is empty list, skip this user
            else:
                nlg = usersNLG[usersNLG['userID'] == userID]['nlg'].unique()[0]
                problemLevelFeatureFrame.iat[i, rewardColumnLocation] = nlg

    problemLevelFeatureFrame.to_csv("./data/with_delayed_rewards/{}/problem-action+reward.csv".format(semester),
                                    index=False)

    # Step-Level Training Data with action column
    step_features = pd.read_csv('data/studies/{}/features_all.csv'.format(semester),
                                header=0, usecols=basic_columns + STEP_FEATURES)

    for problem in problems:
        stepLevelFeatureFrame = step_features[(step_features['decisionPoint'].isin(['stepStart', 'probEnd'])) & (
            step_features['problem'].isin([problem, problem + 'w']))]
        stepLevelFeatureFrame["action"] = ""
        stepLevelFeatureFrame["reward"] = ""
        actionColumnLocation = stepLevelFeatureFrame.columns.get_loc('action')
        stepLevelSubStepFrame = subSteps[subSteps['problem'].isin([problem, problem + 'w'])]

        # make the user IDs consistent with each other
        year_int, semester_int = make_year_and_semester_int(semester)

        if any(stepLevelFeatureFrame.userID < 1000):
            stepLevelFeatureFrame.userID += year_int + semester_int
        if any(stepLevelSubStepFrame.userID < 1000):
            stepLevelSubStepFrame.userID += year_int + semester_int

        # select the minimal set of user IDs to make the data agree with each other
        userIDs = set(stepLevelFeatureFrame.userID.unique()).intersection(stepLevelSubStepFrame.userID.unique())
        # user IDs below 100 are considered test users, remove them
        userIDs = [userID for userID in userIDs if userID > minimum_id(semester)]
        userCount = len(userIDs)

        if semester == 'F16':
            print()

        stepLevelFeatureFrame = stepLevelFeatureFrame[stepLevelFeatureFrame.userID.isin(userIDs)]
        stepLevelSubStepFrame = stepLevelSubStepFrame[stepLevelSubStepFrame.userID.isin(userIDs)]

        # Feature Frame has an extra "probEnd" per problem per user.
        # Since we are at the current problem, it has an additional #NumberofUser rows
        if len(stepLevelFeatureFrame) != len(stepLevelSubStepFrame) + userCount:
            print("feature_all and substep_info length mismatch at step-level")
            continue
            # sys.exit(1)

        subStepCounter = 0
        for i in range(len(stepLevelFeatureFrame)):
            decisionPoint = stepLevelFeatureFrame.iloc[i]['decisionPoint']
            if decisionPoint != 'stepStart':
                continue
            userIDFeature = stepLevelFeatureFrame.iloc[i]['userID']
            userIDSubStep = stepLevelSubStepFrame.iloc[subStepCounter]['userID']

            if userIDFeature != userIDSubStep:
                print("UserID mismatch between feature_all and substep_info at step-level: Feature -> " + str(
                    userIDFeature) + ", subStep -> " + str(userIDSubStep))
                continue
                # sys.exit(1)

            stepLevelFeatureFrame.iat[i, actionColumnLocation] = stepLevelSubStepFrame.iloc[
                subStepCounter]['substepMode']
            subStepCounter += 1

        stepLevelFeatureFrame.to_csv("./data/with_delayed_rewards/{}/".format(semester)
                                     + problem + "(w)-action+reward.csv", index=False)
