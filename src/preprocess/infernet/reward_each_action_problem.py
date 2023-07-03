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


def main(df, action):
    tf.keras.backend.set_floatx("float64")

    if action == 0:
        df["action"] = "problem"
    elif action == 1:
        df["action"] = "example"
    elif action == 2:
        df["action"] = "step_decision"

    max_len = 12
    num_state_features = 130
    num_actions = 3

    def my_loss(y_true, y_pred):
        return K.mean(K.square(K.sum(y_pred, axis=1) - y_true), axis=-1)

    infer_reward_model_path = (
        "model/model_features_all_prob_action_immediate_reward_1000000.h5"
    )
    model = load_model(infer_reward_model_path, custom_objects={"my_loss": my_loss})

    userIDs = df["userID"].unique()

    # Train Infer Net.
    infer_buffer = []
    for user in userIDs:
        data_st = df[df["userID"] == user]
        feats = data_st.iloc[:, 14:]
        non_feats = data_st.iloc[:, :14]
        actions = data_st["action"].tolist()[:]
        actions_ps = np.array([1.0 if x == "problem" else 0.0 for x in actions])
        actions_we = np.array([1.0 if x == "example" else 0.0 for x in actions])
        actions_fwe = np.array(
            [1.0 if x != "problem" and x != "example" else 0.0 for x in actions]
        )
        feats["action_ps"] = actions_ps
        feats["action_we"] = actions_we
        feats["action_fwe"] = actions_fwe
        rewards = data_st["reward"].tolist()
        imm_rews = rewards[:-1]
        feats_np = feats.values
        infer_buffer.append((feats_np, non_feats, imm_rews, rewards[-1], len(rewards)))

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
    if action == -1:
        infer_str = "inferred_rew"
    else:
        infer_str = "inferred_rew_action{}".format(action)
    result_df = pd.DataFrame(
        result,
        columns=df.columns.tolist()
        + ["action_ps", "action_we", "action_fwe", infer_str],
    )
    print("done action{}".format(action))
    return result_df


if __name__ == "__main__":
    df = pd.read_csv(
        "results/nn_inferred_features_all_prob_action_immediate_reward_1000000.csv",
        header=0,
    )
    del df["inferred_rew"]
    del df["action_ps"]
    del df["action_we"]
    del df["action_fwe"]
    actions = df["action"].copy()
    # actions_ps = np.array([1.0 if x == 'problem' else 0.0 for x in actions])
    # actions_we = np.array([1.0 if x == 'example' else 0.0 for x in actions])
    # actions_fwe = np.array([1.0 if x != 'problem' and x != 'example' else 0.0 for x in actions])
    # df['action_ps'] = actions_ps
    # df['action_we'] = actions_we
    # df['action_fwe'] = actions_fwe
    #
    # action_ps = df['action_ps'].copy()
    # action_we = df['action_we'].copy()
    # action_fwe = df['action_fwe'].copy()

    res = main(df, -1)

    lst = []
    for action in range(3):
        infer = main(df, action)
        lst.append(infer)

    res["inferred_rew_action_ps"] = lst[0]["inferred_rew_action0"]
    res["inferred_rew_action_we"] = lst[1]["inferred_rew_action1"]
    res["inferred_rew_action_fwe"] = lst[2]["inferred_rew_action2"]

    df["action"] = actions

    res.to_csv(
        "results/nn_inferred_features_all_prob_action_immediate_reward_all_action.csv",
        index=False,
    )
