import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import numpy as np
import pandas
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from collections import deque
from sklearn.metrics import mean_squared_error
from constant import STEP_FEATURES
import time

tf.random.set_seed(7)

# return the dataset as sample of traces: <student, s, a, r, done>
def getTrace(filename):

    raw_data = pandas.read_csv(filename)
    feature_list = STEP_FEATURES
    feature_len = len(feature_list)

    trace = []

    student_list = list(raw_data['userID'].unique())
    for student in student_list:
        student_data = raw_data.loc[raw_data['userID'] == student,]
        row_index = student_data.index.tolist()


        for i in range(0, len(row_index)):

            state1 = student_data.loc[row_index[i], feature_list].values
            action_type = student_data.loc[row_index[i], 'action']

            if action_type == 'problem':
                action = 0
            else:
                action = 1

            reward = student_data.loc[row_index[i], 'inferred_rew'] * 100

            Done = False
            if (i == len(row_index) - 1):  # the last row is terminal state.
                Done = True
                state2 = np.zeros(feature_len)
            else:
                state2 = student_data.loc[row_index[i+1], feature_list].values

            state1 = np.asarray(state1).astype(np.float64)
            state2 = np.asarray(state2).astype(np.float64)
            trace.append([state1, action, reward, state2, Done])

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


def calculateTarget(model, traces, student_state):
    targetY = []

    predicted_Q = []
    target_Q = []

    targets = model.predict(student_state)

    for i, (state1, action, reward, state2, done) in enumerate(traces):

        predicted_Q.append(targets[i][action])

        if done:
            targets[i][action] = reward
        else:
            Q_future = max(targets[i+1])
            targets[i][action] = reward + Q_future

        target_Q.append(targets[i][action])

        targetY.append(targets[i])

    rmse = mean_squared_error(predicted_Q, target_Q)
    targetY = np.reshape(targetY, (len(targetY), 2))

    return targetY, rmse


def initial_target(traces):
    targetY = []

    for idx, (state1, action, reward, state2, done) in enumerate(traces):
        if action == 0:
            rewards = np.asarray([reward, 0])
        else:
            rewards = np.asarray([0, reward])

        targetY.append(rewards)

    targetY = np.reshape(targetY, (len(targetY), 2))
    return targetY


def main(file_name):
    tf.keras.backend.set_floatx('float64')
    data_path = './new_training_data/nn_inferred_{}.csv'.format(file_name)
    traces, feature_len = getTrace(data_path)


    student_state = []
    for state1, action, reward, state2, done in traces:
        student_state.append(state1)
    student_state = np.asarray(student_state)

    targetY = initial_target(traces)

    print('Start Training')

    for iteration in range(71):

        model = buildModel_LSTM(feature_len)

        t1 = time.time()
        model.fit(student_state, targetY, epochs=50, batch_size=20, shuffle=True, verbose=0)


        directory = 'model/{}/'.format(file_name)
        if not os.path.exists(directory):
            os.makedirs(directory)

        name_problem = directory + "nn_model_{}_{}.h5".format(file_name, iteration)
        model.save(name_problem)

        Q_pred_np_problem = targetY
        max_arg_problem = np.argmax(Q_pred_np_problem, axis=1)
        print("policy for iteration: ", iteration)
        print("step level policy: ", np.unique(max_arg_problem, return_counts=True))


        targetY, RMSE = calculateTarget(model, traces, student_state)
        print("training time is", time.time() - t1, "seconds")
        print("RMSE: ", str(RMSE))

    print('done')

if __name__ == "__main__":

    file_names1 = ['features_all_ex252a', 'features_all_ex212', 'features_all_ex152a', 'features_all_ex242']

    file_names2 = ['features_all_exc137', 'features_all_ex132a', 'features_all_ex132', 'features_all_ex252']

    file_names3 = ['features_all_ex152b', 'features_all_ex152']

    # file_name = file_names2[1]
    for file_name in file_names3:
        print(file_name)
        # file_name = file_names2[1]
        main(file_name)