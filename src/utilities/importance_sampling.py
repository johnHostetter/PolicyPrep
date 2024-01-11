"""
This module implements the ImportanceSampling class, which helps us conduct offline off-policy
evaluation to estimate just how well our trained policies may perform in experiments.
"""
import math


class ImportanceSampling(object):
    """
    Implements multiple ways of calculating the importance sampling.
    """

    def __init__(self, raw_data, theta, gamma, policy):
        self.raw_data = raw_data
        self.theta = theta
        self.gamma = gamma
        self.traces = []
        self.n_action = 0
        self.n_user = 0
        self.random_prob = 0
        self.policy = policy
        self.alpha = 0.5

    def readData(self, problem_id: str):
        raw_data = self.raw_data
        if "problem" in problem_id:
            Q_list = ["ps", "fwe", "we"]
        else:
            Q_list = ["ps", "we"]
        user_list = list(raw_data["userID"].unique())
        self.n_action = len(Q_list)
        self.n_user = len(user_list)
        self.random_prob = 1.0 / self.n_action

        for user in user_list:
            user_sequence = []
            user_data = raw_data.loc[raw_data["userID"] == user,]
            row_index = user_data.index.tolist()

            expert_count = 0
            for i in range(0, len(row_index)):
                action = user_data.loc[row_index[i], "real_action"]
                reward = user_data.loc[row_index[i], "reward"]
                Qs = user_data.loc[row_index[i], Q_list].tolist()

                user_sequence.append((action, reward, Qs))

            self.traces.append(user_sequence)

    def IS(self):
        IS = 0

        for each_student_data in self.traces:
            cumul_policy_prob = 1
            cumul_random_prob = 1
            cumulative_reward = 0

            for i, (action, reward, Qs) in enumerate(each_student_data):
                Q_act = Qs[action]
                prob_logP = math.exp(Q_act * self.theta) / sum(
                    math.exp(x * self.theta) for x in Qs
                )

                cumul_policy_prob *= prob_logP
                cumul_random_prob *= self.random_prob
                cumulative_reward += math.pow(self.gamma, i) * reward

            weight = cumul_policy_prob / cumul_random_prob

            IS_reward = cumulative_reward * weight

            IS += IS_reward

        IS = float(IS) / self.n_user
        return IS

    def WIS(self):
        WIS = 0
        total_weight = 0

        for each_student_data in self.traces:
            cumul_policy_prob = 1
            cumul_random_prob = 1
            cumulative_reward = 0

            for i, (action, reward, Qs) in enumerate(each_student_data):
                Q_act = Qs[action]
                prob_logP = math.exp(Q_act * self.theta) / sum(
                    math.exp(x * self.theta) for x in Qs
                )

                cumul_policy_prob *= prob_logP
                cumul_random_prob *= self.random_prob
                cumulative_reward += math.pow(self.gamma, i) * reward

            weight = cumul_policy_prob / cumul_random_prob

            total_weight += weight
            IS_reward = cumulative_reward * weight

            WIS += IS_reward

        WIS = float(WIS) / total_weight
        return WIS

    def PDIS(self):
        PDIS = 0

        for each_student_data in self.traces:
            cumul_policy_prob = 1
            cumul_random_prob = 1
            PDIS_each_student = 0

            for i, (action, reward, Qs) in enumerate(each_student_data):
                Q_act = Qs[action]
                prob_logP = math.exp(Q_act * self.theta) / sum(
                    math.exp(x * self.theta) for x in Qs
                )

                cumul_policy_prob *= prob_logP
                cumul_random_prob *= self.random_prob
                weight = cumul_policy_prob / cumul_random_prob

                PDIS_each_student += math.pow(self.gamma, i) * reward * weight

            PDIS += PDIS_each_student

        PDIS = float(PDIS) / self.n_user
        return PDIS

    def PDIS2(self):
        PDIS = 0

        for each_student_data in self.traces:
            cumul_policy_prob = 1
            cumul_random_prob = 1
            PDIS_each_student = 0

            for i, (action, reward, Qs) in enumerate(each_student_data):
                Q_act = Qs[action]
                if Q_act == max(Qs):
                    prob_logP = 0.4
                else:
                    prob_logP = 0.2

                # prob_logP = math.exp(Q_act*self.theta) / sum(math.exp(x*self.theta) for x in Qs)

                cumul_policy_prob *= prob_logP
                cumul_random_prob *= self.random_prob
                weight = cumul_policy_prob / cumul_random_prob

                PDIS_each_student += math.pow(self.gamma, i) * reward * weight

            PDIS += PDIS_each_student

        PDIS = float(PDIS) / self.n_user
        return PDIS

    # PHWIS-Behavior
    def PHWIS(self):
        PHWIS_beh = {}
        total_weight = {}
        len_traj = {}
        count_traj = 0

        for each_student_data in self.traces:
            tau = len(each_student_data)
            if len(each_student_data) in len_traj:
                len_traj[tau] += 1
            else:
                len_traj[tau] = 1
                total_weight[tau] = 0
                PHWIS_beh[tau] = 0

            count_traj += 1  # total number of trajectories
            cumul_policy_prob = 1
            cumul_random_prob = 1
            cumulative_reward = 0

            for i, (action, reward, Qs) in enumerate(each_student_data):
                Q_act = Qs[action]
                prob_logP = math.exp(Q_act * self.theta) / sum(
                    math.exp(x * self.theta) for x in Qs
                )

                cumul_policy_prob *= prob_logP
                cumul_random_prob *= self.random_prob
                cumulative_reward += math.pow(self.gamma, i) * reward

            weight = cumul_policy_prob / cumul_random_prob

            total_weight[tau] += weight
            IS_reward = cumulative_reward * weight

            PHWIS_beh[tau] += IS_reward

        PHWIS_beh = {
            tau: float(PHWIS_beh[tau]) / total_weight[tau] for tau in PHWIS_beh
        }
        PHWIS_beh_total = sum(
            (len_traj[tau] / count_traj) * PHWIS_beh[tau] for tau in PHWIS_beh
        )
        return PHWIS_beh_total

    def DR(self):
        DR = 0

        for each_student_data in self.traces:
            cumul_policy_prob = 1
            cumul_random_prob = 1
            DR_each_student = 0
            previous_weight = 1

            for i, (action, reward, Qs) in enumerate(each_student_data):
                Q_act = Qs[action]
                V = max(Qs)
                prob_logP = math.exp(Q_act * self.theta) / sum(
                    math.exp(x * self.theta) for x in Qs
                )

                cumul_policy_prob *= prob_logP
                cumul_random_prob *= self.random_prob
                weight = cumul_policy_prob / cumul_random_prob

                DR_each_student += math.pow(self.gamma, i) * (
                    reward * weight - Q_act * weight + V * previous_weight
                )

                previous_weight = weight

            DR += DR_each_student

        return float(DR) / self.n_user

    def WDR(self):
        DR = 0
        total_weight = 0

        for each_student_data in self.traces:
            cumul_policy_prob = 1
            cumul_random_prob = 1
            DR_each_student = 0
            previous_weight = 1

            for i, (action, reward, Qs) in enumerate(each_student_data):
                Q_act = Qs[action]
                V = max(Qs)
                prob_logP = math.exp(Q_act * self.theta) / sum(
                    math.exp(x * self.theta) for x in Qs
                )

                cumul_policy_prob *= prob_logP
                cumul_random_prob *= self.random_prob
                weight = cumul_policy_prob / cumul_random_prob

                DR_each_student += math.pow(self.gamma, i) * (
                    reward * weight - Q_act * weight + V * previous_weight
                )

                previous_weight = weight

            each_weight = cumul_policy_prob / cumul_random_prob
            total_weight += each_weight

            DR += DR_each_student

        return float(DR) / total_weight
