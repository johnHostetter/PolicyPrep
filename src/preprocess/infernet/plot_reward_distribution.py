import tensorflow as tf
import numpy as np
import pandas as pd
import random
import seaborn as sns
import matplotlib.pyplot as plt


def read_data(file_name):
    data_path = "./training_data_nn/{}.csv".format(file_name)
    data = pd.read_csv(data_path, header=0)
    data = data[data["userID"] > 161000]
    return data


def main():
    data = pd.read_csv(
        "results/nn_inferred_features_all_prob_action_immediate_reward_1000000.csv",
        header=0,
    )

    data = data[["action", "inferred_rew"]].values

    ps = []
    we = []
    fwe = []

    for action, reward in data:
        if action == "problem":
            ps.append(reward)
        elif action == "example":
            we.append(reward)
        else:
            fwe.append(reward)

    print(np.mean(ps))
    print(np.mean(we))
    print(np.mean(fwe))

    kwargs = dict(hist_kws={"alpha": 0.6}, kde_kws={"linewidth": 2})

    plt.figure(figsize=(10, 7), dpi=80)
    sns.distplot(ps, color="dodgerblue", label="ps", **kwargs)
    sns.distplot(we, color="orange", label="we", **kwargs)
    sns.distplot(fwe, color="deeppink", label="fwe", **kwargs)
    plt.xlim(-0.1, 0.1)
    plt.legend()
    plt.savefig("20f_mac.png")

    print("done")


if __name__ == "__main__":
    main()
