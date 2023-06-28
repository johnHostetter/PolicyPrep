import pandas as pd 
pd.options.mode.chained_assignment = None
import math
import numpy as np
import sys

basic_columns = ['recordID', 'answerID', 'time', 'userID', 'problem', 'decisionID', 'decisionPoint', 'decisionOrdering', 'substepMode',	'KC', 'session', 'substepOrdering']

PROBLEM_FEATURES = ['ntellsSinceElicit', 'ntellsSinceElicitKC', 'nElicitSinceTell', 'nElicitSinceTellKC', 'pctElicit',
			   'pctElicitKC', 'pctElicitSession', 'pctElicitKCSession', 'nTellSession', 'nTellKCSession', 'kcOrdering',
			   'kcOrderingSession', 'durationKCBetweenDecision', 'timeInSession', 'timeBetweenSession',
			   'timeOnTutoring',
			   'timeOnTutoringWE', 'timeOnTutoringPS', 'timeOnTutoringKC', 'timeOnTutoringKCWE', 'timeOnTutoringKCPS',
			   'avgTimeOnStep', 'avgTimeOnStepWE', 'avgTimeOnStepPS', 'avgTimeOnStepKC', 'avgTimeOnStepKCWE',
			   'avgTimeOnStepKCPS', 'timeOnLastStepKCPS', 'timeOnTutoringSession', 'timeOnTutoringSessionWE',
			   'timeOnTutoringSessionPS', 'avgTimeOnStepSession', 'avgTimeOnStepSessionWE', 'avgTimeOnStepSessionPS',
			   'nTotalHint', 'nTotalHintSession', 'nHintKC', 'nHintSessionKC', 'AvgTimeOnStepWithHint',
			   'durationSinceLastHint', 'stepsSinceLastHint', 'stepsSinceLastHintKC', 'totalTimeStepsHint',
			   'totalStepsHint',
			   'earlyTraining', 'simpleProblem', 'nKCs', 'nKCsAsPS', 'nKCsSession', 'nKCsSessionPS',
			   'newLevelDifficulty',
			   'nPrincipleInProblem', 'quantitativeDegree', 'nTutorConceptsSession', 'tutAverageConcepts',
			   'tutAverageConceptsSession', 'tutConceptsToWords', 'tutConceptsToWordsSession', 'tutAverageWords',
			   'tutAverageWordsSession', 'tutAverageWordsPS', 'tutAverageWordsSessionPS', 'nDistinctPrincipleInSession',
			   'nPrincipleInSession', 'problemDifficuly', 'problemCompexity', 'problemCategory', 'pctCorrect',
			   'pctOverallCorrect', 'nCorrectKC', 'nIncorrectKC', 'pctCorrectKC', 'pctOverallCorrectKC',
			   'nCorrectKCSession',
			   'nIncorrectKCSession', 'pctCorrectSession', 'pctCorrectKCSession', 'pctOverallCorrectSession',
			   'pctOverallCorrectKCSession', 'nStepSinceLastWrong', 'nStepSinceLastWrongKC', 'nWEStepSinceLastWrong',
			   'nWEStepSinceLastWrongKC', 'nStepSinceLastWrongSession', 'nStepSinceLastWrongKCSession',
			   'nWEStepSinceLastWrongSession', 'nWEStepSinceLastWrongKCSession', 'timeSinceLastWrongStepKC',
			   'nCorrectPSStepSinceLastWrong', 'nCorrectPSStepSinceLastWrongKC',
			   'nCorrectPSStepSinceLastWrongKCSession',
			   'pctCorrectPrin', 'pctCorrectPrinSession', 'nStepSinceLastWrongPrin', 'nWEStepSinceLastWrongPrin',
			   'nStepSinceLastWrongPrinSession', 'nWEStepSinceLastWrongPrinSession', 'nCorrectPSStepSinceLastWrongPrin',
			   'nCorrectPSStepSinceLastWrongPrinSession', 'pctCorrectFirst', 'nStepsSinceLastWrongFirst',
			   'nWEStepSinceLastWrongFirst', 'nCorrectPSStepSinceLastWrongFirst', 'pctCorrectLastProb',
			   'pctCorrectLastProbPrin', 'pctCorrectAdd2Select', 'pctCorrectAdd3Select', 'pctCorrectCompSelect',
			   'pctCorrectDeMorSelect', 'pctCorrectIndeSelect', 'pctCorrectMutualSelect', 'pctCorrectAdd2Apply',
			   'pctCorrectAdd3Apply', 'pctCorrectCompApply', 'pctCorrectDeMorApply', 'pctCorrectIndeApply',
			   'pctCorrectMutualApply', 'pctCorrectAdd2All', 'pctCorrectAdd3All', 'pctCorrectCompAll',
			   'pctCorrectDeMorAll',
			   'pctCorrectIndeAll', 'pctCorrectMutualAll', 'pctCorrectSelectMain', 'nAdd2Prob', 'nAdd3Prob',
			   'nDeMorProb',
			   'nIndeProb', 'nCompProb', 'nMutualProb']

STEP_FEATURES = ['ntellsSinceElicit', 'ntellsSinceElicitKC', 'nElicitSinceTell', 'nElicitSinceTellKC', 'pctElicit', 'pctElicitKC',
	 'pctElicitSession', 'pctElicitKCSession', 'nTellSession', 'nTellKCSession', 'kcOrdering', 'kcOrderingSession',
	 'kcOrderingPb', 'durationKCBetweenDecision', 'timeInSession', 'timeBetweenSession', 'timeOnCurrentProblem',
	 'timeOnTutoring', 'timeOnTutoringWE', 'timeOnTutoringPS', 'timeOnTutoringKC', 'timeOnTutoringKCWE',
	 'timeOnTutoringKCPS', 'avgTimeOnStep', 'avgTimeOnStepWE', 'avgTimeOnStepPS', 'avgTimeOnStepKC',
	 'avgTimeOnStepKCWE', 'avgTimeOnStepKCPS', 'timeOnLastStepKCPS', 'timeOnTutoringSession', 'timeOnTutoringSessionWE',
	 'timeOnTutoringSessionPS', 'avgTimeOnStepSession', 'avgTimeOnStepSessionWE', 'avgTimeOnStepSessionPS',
	 'timeOnTutoringProblem', 'timeOnTutoringProblemWE', 'timeOnTutoringProblemPS', 'avgTimeOnStepProblem',
	 'avgTimeOnStepProblemWE', 'avgTimeOnStepProblemPS', 'nTotalHint', 'nTotalHintSession', 'nHintKC', 'nHintSessionKC',
	 'nTotalHintProblem', 'AvgTimeOnStepWithHint', 'durationSinceLastHint', 'stepsSinceLastHint',
	 'stepsSinceLastHintKC', 'totalTimeStepsHint', 'totalStepsHint', 'earlyTraining', 'simpleProblem', 'nKCs',
	 'nKCsAsPS', 'nKCsSession', 'nKCsSessionPS', 'newLevelDifficulty', 'performanceDifficulty', 'nPrincipleInProblem',
	 'quantitativeDegree', 'nTutorConceptsSession', 'tutAverageConcepts', 'tutAverageConceptsSession',
	 'tutConceptsToWords', 'tutConceptsToWordsSession', 'tutAverageWords', 'tutAverageWordsSession',
	 'tutAverageWordsPS', 'tutAverageWordsSessionPS', 'nDistinctPrincipleInSession', 'nPrincipleInSession',
	 'principleDifficulty', 'principleCategory', 'problemDifficuly', 'problemCompexity', 'problemCategory',
	 'pctCorrect', 'pctOverallCorrect', 'nCorrectKC', 'nIncorrectKC', 'pctCorrectKC', 'pctOverallCorrectKC',
	 'nCorrectKCSession', 'nIncorrectKCSession', 'pctCorrectSession', 'pctCorrectKCSession', 'pctOverallCorrectSession',
	 'pctOverallCorrectKCSession', 'nStepSinceLastWrong', 'nStepSinceLastWrongKC', 'nWEStepSinceLastWrong',
	 'nWEStepSinceLastWrongKC', 'nStepSinceLastWrongSession', 'nStepSinceLastWrongKCSession',
	 'nWEStepSinceLastWrongSession', 'nWEStepSinceLastWrongKCSession', 'timeSinceLastWrongStepKC',
	 'nCorrectPSStepSinceLastWrong', 'nCorrectPSStepSinceLastWrongKC', 'nCorrectPSStepSinceLastWrongKCSession',
	 'pctCorrectPrin', 'pctCorrectPrinSession', 'nStepSinceLastWrongPrin', 'nWEStepSinceLastWrongPrin',
	 'nStepSinceLastWrongPrinSession', 'nWEStepSinceLastWrongPrinSession', 'nCorrectPSStepSinceLastWrongPrin',
	 'nCorrectPSStepSinceLastWrongPrinSession', 'pctCorrectFirst', 'nStepsSinceLastWrongFirst',
	 'nWEStepSinceLastWrongFirst', 'nCorrectPSStepSinceLastWrongFirst', 'pctCorrectLastProb', 'pctCorrectLastProbPrin',
	 'pctCorrectAdd2Select', 'pctCorrectAdd3Select', 'pctCorrectCompSelect', 'pctCorrectDeMorSelect',
	 'pctCorrectIndeSelect', 'pctCorrectMutualSelect', 'pctCorrectAdd2Apply', 'pctCorrectAdd3Apply',
	 'pctCorrectCompApply', 'pctCorrectDeMorApply', 'pctCorrectIndeApply', 'pctCorrectMutualApply', 'pctCorrectAdd2All',
	 'pctCorrectAdd3All', 'pctCorrectCompAll', 'pctCorrectDeMorAll', 'pctCorrectIndeAll', 'pctCorrectMutualAll',
	 'pctCorrectSelectMain', 'nAdd2Prob', 'nAdd3Prob', 'nDeMorProb', 'nIndeProb', 'nCompProb', 'nMutualProb']



subSteps = pd.read_csv('substep_info_F20_S21.csv', header=0)

usersNLG = pd.read_csv('user_NLG_F20_S21.csv', header=0)

userCount = len(subSteps['userID'].unique())
problems = subSteps['problem'].unique()
problems = [p for p in problems if(p[-1] != 'w' and 'ex222' not in p and 'ex144' not in p)]     #Exclude all wording problems and ex222 and ex144 who won't be trained for step-level anyway




# #Problem-Level Training Data with action and reward columns
features = pd.read_csv('features_all_F20_S21.csv', header=0, usecols = basic_columns+PROBLEM_FEATURES)

problemLevelFeatureFrame = features[features['decisionPoint'].isin(['probStart', 'probEnd'])]
problemLevelFeatureFrame.drop(problemLevelFeatureFrame[(problemLevelFeatureFrame['decisionPoint'] == 'probEnd') & (~problemLevelFeatureFrame['problem'].isin(['ex252', 'ex252w']))].index,  inplace=True)
problemLevelFeatureFrame["action"] = ""
problemLevelFeatureFrame["reward"] = ""
actionColumnLocation = problemLevelFeatureFrame.columns.get_loc('action')
rewardColumnLocation = problemLevelFeatureFrame.columns.get_loc('reward')

for i in range(len(problemLevelFeatureFrame)):
    userID = problemLevelFeatureFrame.iloc[i]['userID']
    decisionPoint = problemLevelFeatureFrame.iloc[i]['decisionPoint']
    if(decisionPoint == 'probStart'):
        problem = problemLevelFeatureFrame.iloc[i]['problem']
        uniqueActions = subSteps[(subSteps['userID'] == userID) & (subSteps['problem'] == problem)]['substepMode'].unique()
        if(len(uniqueActions) == 2):
            problemLevelFeatureFrame.iat[i, actionColumnLocation] = 'step_decision'
        else:
            problemLevelFeatureFrame.iat[i, actionColumnLocation] = uniqueActions[0]
    else:
        nlg = usersNLG[usersNLG['userID'] == userID]['NLG_SR'].unique()[0]
        problemLevelFeatureFrame.iat[i, rewardColumnLocation] = nlg


problemLevelFeatureFrame.to_csv("./output/F20+S21-problemLevel-action+reward.csv",index=False)




#Step-Level Training Data with action column
features = pd.read_csv('features_all_F20_S21.csv', header=0, usecols = basic_columns+STEP_FEATURES)

for problem in problems:

    stepLevelFeatureFrame = features[(features['decisionPoint'].isin(['stepStart', 'probEnd'])) & (features['problem'].isin([problem, problem+'w']))]
    stepLevelFeatureFrame["action"] = ""
    stepLevelFeatureFrame["reward"] = ""
    actionColumnLocation = stepLevelFeatureFrame.columns.get_loc('action')

    stepLevelSubStepFrame = subSteps[subSteps['problem'].isin([problem, problem+'w'])]
    
    #Feature Frame has an extra "probEnd" per problem per user. Since we are at the current problem, it has an additional #NumberofUser rows
    if(len(stepLevelFeatureFrame) != len(stepLevelSubStepFrame) + userCount):
        print("feature_all and substep_info length mismatch at step-level")
        sys.exit(1)
    
    subStepCounter = 0
    for i in range(len(stepLevelFeatureFrame)):
        decisionPoint = stepLevelFeatureFrame.iloc[i]['decisionPoint']
        if(decisionPoint != 'stepStart'):
            continue
        userIDFeature = stepLevelFeatureFrame.iloc[i]['userID']
        userIDSubStep = stepLevelSubStepFrame.iloc[subStepCounter]['userID']

        if(userIDFeature != userIDSubStep):
            print("UserID mismatch between feature_all and substep_info at step-level: Feature -> "+ str(userIDFeature)+ ", subStep -> " + str(userIDSubStep))
            sys.exit(1)
        
        stepLevelFeatureFrame.iat[i, actionColumnLocation] = stepLevelSubStepFrame.iloc[subStepCounter]['substepMode']
        subStepCounter += 1

    stepLevelFeatureFrame.to_csv("./output/F20+S21-"+ problem + "(w)-action+reward.csv",index=False)
    