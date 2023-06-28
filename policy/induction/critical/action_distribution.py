import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow.keras.models import load_model
from constant import PROBLEM_FEATURES
import time

# tf.compat.v1.disable_eager_execution()
tf.random.set_seed(7)



def main():
    tf.keras.backend.set_floatx('float64')

    file_name = 'features_all_prob_action_immediate_reward'
    data_path = '../new_training_data/nn_inferred_{}.csv'.format(file_name)

    raw_data = pd.read_csv(data_path)

    raw_data = raw_data[raw_data['problem'] != 'ex222']
    raw_data = raw_data[raw_data['problem'] != 'ex144']

    feature_list = PROBLEM_FEATURES
    student_state = raw_data[feature_list].values
    print('finish loading data')


    for iteration in range(20,101):
        print(iteration)
        nn_str = '../../server_results/policy_induction/model/features_all_prob_action_immediate_reward/nn_model_features_all_prob_action_immediate_reward_{}.h5'.format(iteration)
        nn_model = load_model(nn_str)
        # t1 = time.time()
        predicted = nn_model.predict(student_state)
        # print("prediction time is", time.time() - t1, "seconds")
        policy_actions = np.argmax(predicted, axis=1)

        print("total: {}".format(np.unique(policy_actions, return_counts=True)))

        max_q_problem = np.amax(predicted, axis=1)
        min_q_problem = np.amin(predicted, axis=1)
        
        diff_action_problem = max_q_problem - min_q_problem
        median_diff = np.median(diff_action_problem)
        print(median_diff)

        print("##################################################")


if __name__ == "__main__":
    main()