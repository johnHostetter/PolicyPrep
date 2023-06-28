import tensorflow as tf
import numpy as np
import pandas as pd
import random
from tensorflow.keras.models import load_model
import time
import tensorflow.keras.backend as K


tf.random.set_seed(7)
random.seed(7)
np.random.seed(7)


def main(df, action, file_name):
    tf.keras.backend.set_floatx('float64')

    if action == 0:
        df['action'] = 'problem'
    elif action == 1:
        df['action'] = 'example'

    userIDs = df['userID'].unique()
    max_len = 0
    for user in userIDs:
        data_st = df[df['userID'] == user]
        if len(data_st) > max_len:
            max_len = len(data_st)


    num_state_features = 142
    num_actions = 2

    def my_loss(y_true, y_pred):
        return K.mean(K.square(K.sum(y_pred, axis=1) - y_true), axis=-1)

    infer_reward_model_path = '../sever_results/infer_reward/model/model_{}_{}.h5'.format(file_name, i)
    model = load_model(infer_reward_model_path, custom_objects={'my_loss': my_loss})

    infer_buffer = []
    for user in userIDs:
        data_st = df[df['userID'] == user]
        feats = data_st.iloc[:, 14:]
        non_feats = data_st.iloc[:, :14]
        actions = data_st['action'].tolist()[:]
        actions_ps = np.array([1.0 if x == 'problem' else 0.0 for x in actions])
        actions_we = np.array([1.0 if x == 'example' else 0.0 for x in actions])
        feats['action_ps'] = actions_ps
        feats['action_we'] = actions_we
        rewards = data_st['reward'].tolist()
        imm_rews = rewards[:-1]

        if len(data_st) < max_len:
            num_rows, num_cols = feats.shape
            zeros = np.zeros((max_len - num_rows, num_cols))
            d = pd.DataFrame(zeros, columns=feats.columns)
            imm_rews.extend([0. for _ in range(max_len - num_rows)])
            feats = feats.append(d, ignore_index=True)

        feats_np = feats.values
        infer_buffer.append((feats_np, non_feats, imm_rews, rewards[-1], len(rewards)))


    # Infer the rewards for the data and save the data.
    result = []
    for st in range(len(infer_buffer)):
        sa, non_feats, imm_rews, imm_rew_sum, length = infer_buffer[st]
        non_feats = np.array(non_feats)
        sa = np.reshape(sa, (1, max_len, 144))
        inf_rews = model.predict(sa)

        sa = np.reshape(sa, (max_len, 144))
        sa = sa[:length]
        inf_rews = np.reshape(inf_rews, (max_len, 1))
        inf_rews = inf_rews[:length]
        inf_rews = np.reshape(inf_rews, (length, 1))
        all_feats = np.concatenate((non_feats, sa, inf_rews), axis=-1)
        for row in all_feats:
            result.append(row)

    result = np.array(result)
    if action == -1:
        infer_str = 'inferred_rew'
    else:
        infer_str = 'inferred_rew_aciton{}'.format(action)

    result_df = pd.DataFrame(result, columns=df.columns.tolist() + ['action_ps', 'action_we', infer_str])

    print('done action{}'.format(action))
    return result_df


if __name__ == '__main__':
    file_names = ['features_all_ex132', 'features_all_ex132a', 'features_all_ex152',
                  'features_all_ex152a', 'features_all_ex152b', 'features_all_ex212',
                  'features_all_ex242', 'features_all_ex252', 'features_all_ex252a'
                  ]

    # file_names = ['features_all_exc137']
    i = 1000000
    for file_name in file_names:
        print(file_name)
        df = pd.read_csv('../sever_results/infer_reward/results/nn_inferred_{}_{}.csv'.format(file_name, i), header=0)
        del df['inferred_rew']
        del df['action_ps']
        del df['action_we']
        actions = df['action'].copy()


        res = main(df, -1, file_name)

        lst = []
        for action in range(2):
            infer = main(df, action, file_name)
            lst.append(infer)

        res['inferred_rew_aciton_ps'] = lst[0]['inferred_rew_aciton0']
        res['inferred_rew_aciton_we'] = lst[1]['inferred_rew_aciton1']

        df['action'] = actions

        res.to_csv('results/nn_inferred_{}_all_action.csv'.format(file_name), index=False)
