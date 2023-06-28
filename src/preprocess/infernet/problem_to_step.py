import pandas as pd
from collections import defaultdict

PROBLEM_LIST = ['exc137', 'ex132a', 'ex132', 'ex152a', 'ex152b', 'ex152', 'ex212', 'ex242', 'ex252a', 'ex252']
def read_data(file_name, user_list):
    data_path = './training_data/{}_immediate_reward.csv'.format(file_name)
    data = pd.read_csv(data_path, header=0)
    data = data[data['userID'].isin(user_list)]
    return data


def main():

    data_path = './results/nn_inferred_features_all_prob_action_immediate_reward_1000000.csv'
    problem_data = pd.read_csv(data_path, header=0)

    user_list = problem_data['userID'].unique()
    userIDs = problem_data.userID.values
    problems = problem_data.problem.values
    infer_rewards = problem_data.inferred_rew.values

    user_problem_reward = defaultdict(dict)
    for i in range(len(userIDs)):
        userID = userIDs[i]
        problem = problems[i]
        infer_reward = infer_rewards[i]
        user_problem_reward[userID][problem] = infer_reward

    file_names = ['features_all_ex132', 'features_all_ex132a', 'features_all_ex152',
                  'features_all_ex152a', 'features_all_ex152b', 'features_all_ex212',
                  'features_all_ex242', 'features_all_ex252',
                  'features_all_ex252a', 'features_all_exc137'
                  ]

    for file_name in file_names:
    # file_name = file_names[9]
        print(file_name)
        step_data = read_data(file_name, user_list)

        userIDs = step_data['userID'].unique()
        for user in userIDs:
            for problem in PROBLEM_LIST:
                nn_inferred_reward = user_problem_reward[user][problem]
                step_data.loc[(step_data.userID == user) & (step_data.problem == problem) & (step_data.decisionPoint == 'probEnd'), 'reward'] = nn_inferred_reward

        step_data.to_csv('training_data_nn/{}_nn_immediate_reward.csv'.format(file_name), index=False)


    print('done')


if __name__ == '__main__':
    main()

