import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import numpy as np
import pandas
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error
from constant import STEP_FEATURES
from constant import MEDIAN_THRESHOLD_STR_POSITIVE
from constant import MEDIAN_THRESHOLD_STR_NEGATIVE
import time

tf.random.set_seed(7)

# return the dataset as sample of traces: <student, s, a, r, done>
def getTrace(path, filename):

    raw_data = pandas.read_csv(path)
    feature_list = STEP_FEATURES
    feature_len = len(feature_list)

    positive_thres = MEDIAN_THRESHOLD_STR_POSITIVE[filename]
    negative_thres = MEDIAN_THRESHOLD_STR_NEGATIVE[filename]

    raw_data['short_critical'] = np.where((raw_data['inferred_rew_aciton_ps'] > positive_thres) | (raw_data['inferred_rew_aciton_we'] > positive_thres) | (raw_data['inferred_rew_aciton_ps'] < negative_thres) | (raw_data['inferred_rew_aciton_we'] < negative_thres), 1, 0)

    trace = []

    student_list = list(raw_data['userID'].unique())
    for student in student_list:
        student_data = raw_data.loc[raw_data['userID'] == student,]
        row_index = student_data.index.tolist()


        for i in range(0, len(row_index)):

            state1 = student_data.loc[row_index[i], feature_list].values
            action_type = student_data.loc[row_index[i], 'action']

            ShortTR2 = False
            if action_type == 'problem':
                action = 0
            else:
                action = 1

            reward = student_data.loc[row_index[i], 'inferred_rew'] * 1000

            Done = False
            if (i == len(row_index) - 1):  # the last row is terminal state.
                Done = True
                state2 = np.zeros(feature_len)
                action2 = 0
            else:
                state2 = student_data.loc[row_index[i+1], feature_list].values
                action_type2 = student_data.loc[row_index[i+1], 'action']
                if action_type2 == 'problem':
                    action2 = 0
                else:
                    action2 = 1

                short_critical = student_data.loc[row_index[i+1], 'short_critical']
                if short_critical == 1:
                    ShortTR2 = True

            state1 = np.asarray(state1).astype(np.float64)
            state2 = np.asarray(state2).astype(np.float64)
            trace.append([state1, action, reward, state2, action2, ShortTR2, Done])

    return trace, feature_len



def buildModel_LSTM(feature_len):

    model = Sequential()
    model.add(Dense(256, input_dim=feature_len, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(2, activation='linear'))
    model.compile(loss='mse', optimizer=Adam(lr=0.001))

    return model


def calculateTarget(model, traces, student_state, q_diff_percentage):
    targetY = []

    predicted_Q = []
    target_Q = []

    targets = model.predict(student_state)

    Q_pred_np_problem = np.reshape(targets, (len(targets), 2))

    max_q_problem = np.amax(Q_pred_np_problem, axis=1)
    min_q_problem = np.amin(Q_pred_np_problem, axis=1)

    diff_action_problem = max_q_problem - min_q_problem

    q_diff_values = diff_action_problem.copy()
    q_diff_sorted = sorted(q_diff_values, key=lambda x: float(x), reverse=True)
    position = int((len(q_diff_sorted) - 1) * q_diff_percentage)
    q_diff_threshold = q_diff_sorted[position]

    # median_diff = np.median(diff_action_problem)
    # print(q_diff_threshold)
    # print(median_diff)
    for i, (state1, action, reward, state2, action2, ShortTR2, done) in enumerate(traces):

        predicted_Q.append(targets[i][action])

        if done:
            targets[i][action] = reward
        else:
            Qs_next = targets[i + 1]
            Q_next_diff = diff_action_problem[i + 1]

            if Q_next_diff > q_diff_threshold or ShortTR2 == True:
                Q_future = max(Qs_next)
            else:
                Q_future = np.mean(Qs_next)

            targets[i][action] = reward + Q_future

        target_Q.append(targets[i][action])

        targetY.append(targets[i])

    rmse = mean_squared_error(predicted_Q, target_Q)
    targetY = np.reshape(targetY, (len(targetY), 2))

    return targetY, rmse


def initial_target(traces):
    targetY = []

    for idx, (state1, action, reward, state2, action2, ShortTR2, done) in enumerate(traces):
        if action == 0:
            rewards = np.asarray([reward, 0])
        else:
            rewards = np.asarray([0, reward])

        targetY.append(rewards)

    targetY = np.reshape(targetY, (len(targetY), 2))
    return targetY


def main(file_name):
    tf.keras.backend.set_floatx('float64')
    q_diff_percentage = 0.5
    data_path = './new_training_data/nn_inferred_features_all_{}_all_action.csv'.format(file_name)
    traces, feature_len = getTrace(data_path, file_name)


    student_state = []
    for state1, action, reward, state2, action2, ShortTR2, done in traces:
        student_state.append(state1)
    student_state = np.asarray(student_state)

    targetY = initial_target(traces)

    print('Start Training')


    for iteration in range(61):

        model = buildModel_LSTM(feature_len)

        t1 = time.time()
        model.fit(student_state, targetY, epochs=50, batch_size=20, shuffle=True, verbose=0)

        directory = 'model_modifyBellman/features_all_{}/'.format(file_name)
        if not os.path.exists(directory):
            os.makedirs(directory)

        name_problem = directory + "nn_model_{}_{}.h5".format(file_name, iteration)
        model.save(name_problem)

        Q_pred_np_problem = targetY
        max_arg_problem = np.argmax(Q_pred_np_problem, axis=1)
        print("policy for iteration: ", iteration)
        print("step level policy: ", np.unique(max_arg_problem, return_counts=True))


        targetY, RMSE = calculateTarget(model, traces, student_state, q_diff_percentage)
        print("training time is", time.time() - t1, "seconds")
        print("RMSE: ", str(RMSE))

    print('done')

if __name__ == "__main__":

    file_names_all = ['ex252a', 'ex212', 'ex152a', 'ex242', 'exc137', 'ex132a', 'ex132', 'ex252', 'ex152b', 'ex152']

    file_names1 = ['ex252a', 'ex212', 'ex152a', 'ex242']

    file_names2 = ['exc137', 'ex132a', 'ex132', 'ex252']

    file_names3 = ['ex152b', 'ex152']


    for file_name in file_names1:
        print(file_name)
        main(file_name)