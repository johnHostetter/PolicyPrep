import math
import pandas as pd
import numpy as np

MEDIAN_THRESHOLD_STR_POSITIVE = {
	'problem': 0.0659,
    'exc137': 0.0013,
    'ex132a': 0.00132,
    'ex132': 0.00156,
    'ex152a': 0.00173,
    'ex152b': 0.00143,
    'ex152': 0.00133,
    'ex212': 0.00348,
    'ex242': 0.00295,
    'ex252a': 0.00675,
    'ex252': 0.00175
}

MEDIAN_THRESHOLD_STR_NEGATIVE = {
	'problem': -0.0856,
    'exc137': -0.0008,
    'ex132a': -0.00113,
    'ex132': -0.00155,
    'ex152a': -0.00188,
    'ex152b': -0.00163,
    'ex152': -0.0015,
    'ex212': -0.00435,
    'ex242': -0.0036,
    'ex252a': -0.009,
    'ex252': -0.00237
}

def main():
    #
    # for problem in ['exc137', 'ex132a', 'ex132', 'ex152a', 'ex152b', 'ex152', 'ex212', 'ex242', 'ex252a', 'ex252']:
    #     print(problem)
    #     file_name = 'features_all_{}'.format(problem)
    #     data_path = 'new_training_data/nn_inferred_{}_all_action.csv'.format(file_name)
    #
    #     # data_path = "new_training_data/nn_inferred_features_all_prob_action_immediate_reward_all_action.csv"
    #
    #     positive_thres = MEDIAN_THRESHOLD_STR_POSITIVE[problem]
    #     negative_thres = MEDIAN_THRESHOLD_STR_NEGATIVE[problem]
    #
    #     raw_data = pd.read_csv(data_path)
    #     raw_data['total_critical'] = np.where((raw_data['inferred_rew_aciton_ps'] > positive_thres) | (raw_data['inferred_rew_aciton_we'] > positive_thres) | (raw_data['inferred_rew_aciton_ps'] < negative_thres) | (raw_data['inferred_rew_aciton_we'] < negative_thres), 1, 0)
    #     raw_data['positive_critical'] = np.where(
    #         (raw_data['inferred_rew_aciton_ps'] > positive_thres) | (raw_data['inferred_rew_aciton_we'] > positive_thres), 1, 0)
    #
    #     raw_data['negative_critical'] = np.where((raw_data['inferred_rew_aciton_ps'] < negative_thres) | (
    #                     raw_data['inferred_rew_aciton_we'] < negative_thres), 1, 0)
    #
    #     total_len = len(raw_data)
    #     critical_len = len(raw_data[raw_data['total_critical'] == 1])
    #     positive = len(raw_data[raw_data['positive_critical'] == 1])
    #     negative = len(raw_data[raw_data['negative_critical'] == 1])
    #     print("{},{}".format(critical_len, critical_len/total_len))
    #     print("{},{}".format(positive, positive/total_len))
    #     print("{},{}".format(negative, negative/total_len))

        # print('done')


    data_path = 'new_training_data/nn_inferred_features_all_prob_action_immediate_reward_all_action.csv'

    positive_thres = MEDIAN_THRESHOLD_STR_POSITIVE['problem']
    negative_thres = MEDIAN_THRESHOLD_STR_NEGATIVE['problem']

    raw_data = pd.read_csv(data_path)
    raw_data['total_critical'] = np.where((raw_data['inferred_rew_aciton_ps'] > positive_thres) | (raw_data['inferred_rew_aciton_we'] > positive_thres) | (raw_data['inferred_rew_aciton_fwe'] > positive_thres) | (raw_data['inferred_rew_aciton_ps'] < negative_thres) | (raw_data['inferred_rew_aciton_we'] < negative_thres) | (raw_data['inferred_rew_aciton_fwe'] < negative_thres), 1, 0)
    raw_data['positive_critical'] = np.where(
        (raw_data['inferred_rew_aciton_ps'] > positive_thres) | (raw_data['inferred_rew_aciton_we'] > positive_thres) | (
                    raw_data['inferred_rew_aciton_fwe'] > positive_thres), 1, 0)

    raw_data['negative_critical'] = np.where((raw_data['inferred_rew_aciton_ps'] < negative_thres) | (
                    raw_data['inferred_rew_aciton_we'] < negative_thres) | (
                    raw_data['inferred_rew_aciton_fwe'] < negative_thres), 1, 0)

    total_len = len(raw_data)
    critical_len = len(raw_data[raw_data['total_critical'] == 1])
    positive = len(raw_data[raw_data['positive_critical'] == 1])
    negative = len(raw_data[raw_data['negative_critical'] == 1])
    print("{},{}".format(critical_len, critical_len/total_len))
    print("{},{}".format(positive, positive/total_len))
    print("{},{}".format(negative, negative/total_len))

    print('done')


if __name__ == "__main__":
    main()


