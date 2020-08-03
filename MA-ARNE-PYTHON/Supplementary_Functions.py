import numpy as np

class Supp_functions:
    num_states = None
    entry_dest_info = None

    def __init__(self, num_states, entry_dest_info):
        self.num_states = num_states
        self.entry_dest_info = entry_dest_info

    def act_choice(self, Prob_dist):
        chosen_act_k = 0
        num_act_k = len(Prob_dist)
        if num_act_k > 0:
            P_temp = sorted(Prob_dist)
            I = sorted(range(len(P_temp)), key=lambda k: P_temp[k])
            P = []

            for jj in range(0, num_act_k):
                P.append(sum(P_temp[0:jj+1]))
            choice = np.random.uniform(0, 1, 1)

            for ll in range(0, num_act_k):
                if ll == 0:
                    lb = 0
                else:
                    lb = P[ll - 1]
                if (lb <= choice) and (choice <= P[ll]):
                    chosen_act_k = I[ll]

        return chosen_act_k

    def policy_projection(self, policy):
        u = sorted(policy, reverse=True)
        max_index = 0
        find_max_index_flag = 0

        while find_max_index_flag == 0:
            temp_max_val = u[max_index] + (1 / (max_index + 1))*(1 - sum(u[0:max_index+1]))
            if temp_max_val > 0:
                max_index = max_index + 1
            else:
                find_max_index_flag = 1

            if max_index == len(policy):
                find_max_index_flag = 1

        lambda_val = (1 / max_index)*(1 - sum(u[0:max_index]))
        p = np.zeros(len(policy))
        for ii in range(0, len(policy)):
            p[ii] = max([policy[ii] + lambda_val, 0])

        return p.tolist()

    def random_initial_state(self, State_Tracker):

        avoid_states = []
        state_space = list(range(0, self.num_states))
        for ii in range(1, len(self.entry_dest_info)):  # Collect all destination related states
            avoid_states.extend(self.entry_dest_info[ii])

        avoid_states.extend([len(state_space)-1])  # Avoid choosing s_0 as next state

        flag = 0
        sorted_indices = sorted(range(len(State_Tracker)), key=lambda k: State_Tracker[k])
        counter = 0
        state = []

        while flag == 0:
            state = state_space[sorted_indices[counter]]

            avoid_flag = state in avoid_states
            if int(avoid_flag) == 0:  # Chosen state is not a avoid state
                flag = 1

            counter = counter + 1

        return state
