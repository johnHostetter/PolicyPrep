import tensorflow as tf
import numpy as np
import pandas as pd
import random
from tensorflow.keras import Sequential
from tensorflow.keras.layers import TimeDistributed, Dense, Dropout, LeakyReLU
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import Adam
import time

tf.random.set_seed(7)
random.seed(7)
np.random.seed(7)


def read_data(file_name):
    data_path = './training_data/{}.csv'.format(file_name)
    data = pd.read_csv(data_path, header=0)
    data = data[data['userID'] > 161000]
    return data


def model_build(max_ep_length, num_sas_features):
    model = Sequential()
    model.add(TimeDistributed(Dense(256), input_shape=(max_ep_length, num_sas_features)))
    model.add(LeakyReLU())
    model.add(Dropout(0.5))
    model.add(TimeDistributed(Dense(256)))
    model.add(LeakyReLU())
    model.add(Dropout(0.5))
    model.add(TimeDistributed(Dense(256)))
    model.add(LeakyReLU())
    model.add(TimeDistributed(Dense(1)))

    def my_loss(y_true, y_pred):
        inferred_sum = K.sum(y_pred, axis=1)
        inferred_sum = tf.reshape(inferred_sum, (tf.shape(y_true)[0], tf.shape(y_true)[1]))
        return K.mean(K.square(inferred_sum - y_true), axis=-1)

    model.compile(loss=my_loss, optimizer=Adam(lr=0.0001))

    return model


def main():
    tf.keras.backend.set_floatx('float64')
    file_name = 'features_all_prob_action_immediate_reward'
    data_original = read_data(file_name)
    userIDs = data_original['userID'].unique()
    max_len = 0
    for user in userIDs:
        data_st = data_original[data_original['userID'] == user]
        if len(data_st) > max_len:
            max_len = len(data_st)
    max_len -= 1


    num_state_features = 130
    num_actions = 3
    print(file_name)
    print('Max episode length is {}'.format(max_len))

    # Normalize each column.
    data = data_original.copy()
    feats, mins, maxs = [], [], []
    for feature_name in data.columns[14:]:
        max_val = data[feature_name].max()
        min_val = data[feature_name].min()
        if min_val == max_val:
            data[feature_name] = 0.0
        else:
            data[feature_name] = (data_original[feature_name] - min_val) / (max_val - min_val)
        feats.append(feature_name)
        mins.append(min_val)
        maxs.append(max_val)
    df = pd.DataFrame({'feat': feats, 'min_val': mins, 'max_val': maxs})
    df.to_csv('normalization_values/normalization_{}.csv'.format(file_name), index=False)
    # quit()
    
    # Train Infer Net.
    model = model_build(max_len, num_state_features+num_actions)
    infer_buffer = []
    for user in userIDs:
        data_st = data[data['userID'] == user]
        feats = data_st.iloc[:-1, 14:]
        non_feats = data_st.iloc[:-1, :14]
        actions = data_st['action'].tolist()[:-1]
        actions_ps = np.array([1.0 if x == 'problem' else 0.0 for x in actions])
        actions_we = np.array([1.0 if x == 'example' else 0.0 for x in actions])
        actions_fwe = np.array([1.0 if x != 'problem' and x != 'example' else 0.0 for x in actions])
        feats['action_ps'] = actions_ps
        feats['action_we'] = actions_we
        feats['action_fwe'] = actions_fwe
        rewards = data_st['reward'].tolist()
        imm_rews = rewards[:-1]
        feats_np = feats.values
        infer_buffer.append((feats_np, non_feats, imm_rews, rewards[-1], len(rewards)-1))

    # Train infer_net.
    train_steps = 1000001
    infer_batch_size = 20
    print("#####################")
    t1 = time.time()
    losses = []
    for i in range(train_steps):
        batch = random.sample(infer_buffer, infer_batch_size)
        sa, non_feats, imm_rews, imm_rew_sum, length = list(zip(*batch))
        sa = np.reshape(sa, (infer_batch_size, max_len, 133))
        imm_rew_sum = np.reshape(imm_rew_sum, (infer_batch_size, 1))

        hist = model.fit(sa, imm_rew_sum, epochs=1, batch_size=infer_batch_size, verbose=0)
        loss = hist.history['loss'][0]
        losses.append(loss)
        if i == 0:
            print(file_name)
        if i % 1000 == 0:
            print('Step {}/{}, loss {}'.format(i, train_steps, loss))
            print("Training time is", time.time() - t1, "seconds")
            t1 = time.time()

        if i == 500000 or i == 1000000:
            # Infer the rewards for the data and save the data.
            result = []
            for st in range(len(infer_buffer)):
                sa, non_feats, imm_rews, imm_rew_sum, length = infer_buffer[st]
                non_feats = np.array(non_feats)
                sa = np.reshape(sa, (1, max_len, 133))
                inf_rews = model.predict(sa)

                sa = np.reshape(sa, (max_len, 133))
                inf_rews = np.reshape(inf_rews, (length, 1))
                all_feats = np.concatenate((non_feats, sa, inf_rews), axis=-1)
                for row in all_feats:
                    result.append(row)

            result = np.array(result)
            result_df = pd.DataFrame(result, columns=data.columns.tolist() + ['action_ps', 'action_we', 'action_fwe', 'inferred_rew'])
            result_df.to_csv('results/nn_inferred_{}_{}.csv'.format(file_name, i), index = False)


            df = pd.DataFrame({'loss': losses})
            df.to_csv('figures/loss_{}_{}.csv'.format(file_name, i), index=False)

            model.save('model/model_{}_{}.h5'.format(file_name, i))

    print('done')


if __name__ == '__main__':
    main()
