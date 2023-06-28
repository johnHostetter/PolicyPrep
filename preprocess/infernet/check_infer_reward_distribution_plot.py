import pandas as pd
import matplotlib.pyplot as plt


def read_data(file_name):
    data_path = 'results/nn_inferred_{}_1000000.csv'.format(file_name)
    data = pd.read_csv(data_path, header=0)

    return data


def main():

    file_names2 = ['features_all_ex132', 'features_all_ex132a', 'features_all_ex152',
                  'features_all_ex152a', 'features_all_ex152b', 'features_all_ex212',
                  'features_all_ex242', 'features_all_ex252',
                  'features_all_ex252a', 'features_all_exc137'
                  ]

    file_names = ['features_all_prob_action_immediate_reward']
    for file_name in file_names:
        data = read_data(file_name)
        print(file_name)
        inf_rewards = sorted(data.inferred_rew.values, reverse=True)
        length = len(inf_rewards)
        first_pos = int(length*0.04)
        end_pos = int(length*0.96)

        first05 = inf_rewards[first_pos]
        end05 = inf_rewards[end_pos]
        print(first05, end05)

        fig = plt.figure()
        plt.plot(inf_rewards)
        fig.suptitle(file_name, fontsize=20)
        x1, x2, y1, y2 = plt.axis()
        # plt.axis([x1, x2, -0.025, 0.025])
        plt.axis([x1, x2, -0.5, 0.5])

        plt.axvline(x=first_pos)
        plt.axvline(x=end_pos)
        fig.savefig('infer_reward_distribution_figure/{}.png'.format(file_name))


    print('done')


if __name__ == '__main__':
    main()

