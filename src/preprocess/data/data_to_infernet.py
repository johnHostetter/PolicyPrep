"""
This Python module contains the code to generate training data with action and reward columns.
The format of the training data is required for the InferNet code.
"""
import sys

import pandas as pd

from src.utils.reproducibility import load_configuration


if __name__ == "__main__":
    pd.options.mode.chained_assignment = None

    config = load_configuration("default_configuration.yaml")

    USER_COLUMNS = config.data.features.basic + config.data.features.problem

    subSteps = pd.read_csv("data/substep_info.csv", header=0)
    usersNLG = pd.read_csv("user_NLG.csv", header=0)

    userCount = len(subSteps["userID"].unique())
    problems = subSteps["problem"].unique()
    problems = [
        p for p in problems if (p[-1] != "w" and "ex222" not in p and "ex144" not in p)
    ]  # Exclude all wording problems and ex222 and ex144 who won't be trained for step-level anyway

    # #Problem-Level Training Data with action and reward columns
    features = pd.read_csv(
        "data/features_all.csv",
        header=0,
        usecols=config.data.features.basic + config.data.features.problem,
    )

    problemLevelFeatureFrame = features[
        features["decisionPoint"].isin(["probStart", "probEnd"])
    ]
    problemLevelFeatureFrame.drop(
        problemLevelFeatureFrame[
            (problemLevelFeatureFrame["decisionPoint"] == "probEnd")
            & (~problemLevelFeatureFrame["problem"].isin(["ex252", "ex252w"]))
            ].index,
        inplace=True,
    )
    problemLevelFeatureFrame["action"] = ""
    problemLevelFeatureFrame["reward"] = ""
    actionColumnLocation = problemLevelFeatureFrame.columns.get_loc("action")
    rewardColumnLocation = problemLevelFeatureFrame.columns.get_loc("reward")

    for i in range(len(problemLevelFeatureFrame)):
        userID = problemLevelFeatureFrame.iloc[i]["userID"]
        decisionPoint = problemLevelFeatureFrame.iloc[i]["decisionPoint"]
        if decisionPoint == "probStart":
            problem = problemLevelFeatureFrame.iloc[i]["problem"]
            uniqueActions = subSteps[
                (subSteps["userID"] == userID) & (subSteps["problem"] == problem)
                ]["substepMode"].unique()
            if len(uniqueActions) == 2:
                problemLevelFeatureFrame.iat[i, actionColumnLocation] = "step_decision"
            else:
                problemLevelFeatureFrame.iat[i, actionColumnLocation] = uniqueActions[0]
        else:
            nlg = usersNLG[usersNLG["userID"] == userID]["NLG_SR"].unique()[0]
            problemLevelFeatureFrame.iat[i, rewardColumnLocation] = nlg

    problemLevelFeatureFrame.to_csv(
        "./output/problemLevel-action+reward.csv", index=False
    )

    # Step-Level Training Data with action column
    features = pd.read_csv(
        "features_all_F20_S21.csv",
        header=0,
        usecols=config.data.features.basic + config.data.features.step,
    )

    for problem in problems:
        stepLevelFeatureFrame = features[
            (features["decisionPoint"].isin(["stepStart", "probEnd"]))
            & (features["problem"].isin([problem, problem + "w"]))
            ]
        stepLevelFeatureFrame["action"] = ""
        stepLevelFeatureFrame["reward"] = ""
        actionColumnLocation = stepLevelFeatureFrame.columns.get_loc("action")

        stepLevelSubStepFrame = subSteps[subSteps["problem"].isin([problem, problem + "w"])]

        # Feature Frame has an extra "probEnd" per problem per user. Since we are at the current problem, it has an additional #NumberofUser rows
        if len(stepLevelFeatureFrame) != len(stepLevelSubStepFrame) + userCount:
            print("feature_all and substep_info length mismatch at step-level")
            sys.exit(1)

        sub_step_counter = 0
        for i in range(len(stepLevelFeatureFrame)):
            decisionPoint = stepLevelFeatureFrame.iloc[i]["decisionPoint"]
            if decisionPoint != "stepStart":
                continue
            userIDFeature = stepLevelFeatureFrame.iloc[i]["userID"]
            userIDSubStep = stepLevelSubStepFrame.iloc[sub_step_counter]["userID"]

            if userIDFeature != userIDSubStep:
                print(
                    "UserID mismatch between feature_all and substep_info at step-level: Feature -> "
                    + str(userIDFeature)
                    + ", subStep -> "
                    + str(userIDSubStep)
                )
                sys.exit(1)

            stepLevelFeatureFrame.iat[i, actionColumnLocation] = stepLevelSubStepFrame.iloc[
                sub_step_counter
            ]["substepMode"]
            sub_step_counter += 1

        stepLevelFeatureFrame.to_csv(
            "./output/" + problem + "(w)-action+reward.csv", index=False
        )
