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
from constant import PROBLEM_FEATURES
import time

tf.random.set_seed(7)

# return the dataset as sample of traces: <student, s, a, r, done>
def getTrace(filename):

    raw_data = pandas.read_csv(filename)
    feature_list = PROBLEM_FEATURES
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
            elif action_type == 'example':
                action = 2
            else:
                action = 1

            reward = student_data.loc[row_index[i], 'inferred_rew'] * 10

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
    model.add(Dense(3, activation='linear'))
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
    targetY = np.reshape(targetY, (len(targetY), 3))

    return targetY, rmse


def initial_target(traces):
    targetY = []

    for idx, (state1, action, reward, state2, done) in enumerate(traces):
        if action == 0:
            rewards = np.asarray([reward, 0, 0])
        elif action == 1:
            rewards = np.asarray([0, reward, 0])
        else:
            rewards = np.asarray([0, 0, reward])

        # rewards = np.reshape(rewards, (1,3))
        targetY.append(rewards)

    return np.asarray(targetY)


def main():
    tf.keras.backend.set_floatx('float64')
    past_iterations = 5
    policy_fixed_problem = False
    file_name = 'features_all_prob_action_immediate_reward'

    data_path = './new_training_data/nn_inferred_{}.csv'.format(file_name)
    traces, feature_len = getTrace(data_path)


    student_state = []
    for state1, action, reward, state2, done in traces:
        student_state.append(state1)
    student_state = np.asarray(student_state)

    targetY = initial_target(traces)

    print('State training')
    policy_container_problem = deque()

    for iteration in range(101):

        model = buildModel_LSTM(feature_len)

        t1 = time.time()
        model.fit(student_state, targetY, epochs=50, batch_size=20, shuffle=True, verbose=0)


        directory = 'model/{}/'.format(file_name)
        if not os.path.exists(directory):
            os.makedirs(directory)

        name_problem = directory + "nn_model_{}_{}.h5".format(file_name, iteration)
        model.save(name_problem)

        Q_pred_np_problem = np.reshape(targetY, (len(targetY), 3))
        max_arg_problem = np.argmax(Q_pred_np_problem, axis=1)
        print("policy for iteration:", iteration)
        print("problem level policy", np.unique(max_arg_problem, return_counts=True))

        # policy container for comparison
        policy_container_problem.append(max_arg_problem)
        if len(policy_container_problem) > past_iterations:
            policy_container_problem.popleft()
            policy_fixed_problem = (np.diff(np.vstack(policy_container_problem).reshape(len(policy_container_problem), -1), axis=0) == 0).all()

        if policy_fixed_problem:
            break

        targetY, RMSE = calculateTarget(model, traces, student_state)
        print("training time is", time.time() - t1, "seconds")
        print("RMSE: ", str(RMSE))

    print('done')

if __name__ == "__main__":
    main()