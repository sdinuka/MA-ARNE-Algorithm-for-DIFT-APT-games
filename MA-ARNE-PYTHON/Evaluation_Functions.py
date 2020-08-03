from Supplementary_Functions import *

class Eval_and_Plots:
    Policy_Data = None
    t = None
    N_ss = None
    stage_ID = None
    ss_entry_dest = None
    CD = None
    APT_win = None
    APT_drop = None
    DIFT_win = None
    DIFT_lose = None
    trap_set = None
    state_transition_matrix = None
    FN = None
    s0 = None
    V_Data = None

    def __init__(self, Policy_Data, t, N_ss, stage_ID, ss_entry_dest, CD, APT_win, APT_drop, DIFT_win, DIFT_lose, trap_set, state_transition_matrix, FN, s0, V_Data):
        self.Policy_Data = Policy_Data
        self.t = t
        self.N_ss = N_ss
        self.stage_ID = stage_ID
        self.ss_entry_dest = ss_entry_dest
        self.CD = CD
        self.APT_win = APT_win
        self.APT_drop = APT_drop
        self.DIFT_win = DIFT_win
        self.DIFT_lose = DIFT_lose
        self.trap_set = trap_set
        self.state_transition_matrix = state_transition_matrix
        self.FN = FN
        self.s0 = s0
        self.V_Data = V_Data

    def find_average_reward(self, T_sim):
        #T_sim: Time horizon of the DIFT-APT game to calculate the average reward
        Output_Data_average_reward = []
        Avg_reward_DIFT = np.zeros(T_sim+1)
        Avg_reward_APT = np.zeros(T_sim+1)
        action_set_DIFT = np.zeros(T_sim+1)
        action_set_APT = np.zeros(T_sim+1)
        Supp_Func_Obj = Supp_functions(self.N_ss, self.ss_entry_dest)
        current_state = self.s0

        for ii in range(0, T_sim):
            reward_DIFT = 0 #DIFT's reward
            reward_APT = 0 #APT's reward
            current_state_ID = current_state #Finding the state ID

            #Choosing player actions
            #DIFT
            DIFT_action_ID = Supp_Func_Obj.act_choice(self.Policy_Data[self.t][0][current_state_ID])
            #APT
            APT_action_ID = Supp_Func_Obj.act_choice(self.Policy_Data[self.t][1][current_state_ID])

            neighor_IDs = [index for index in range(self.N_ss) if self.state_transition_matrix[current_state_ID][index] == 1]

            #Recording the actions chosen by DIFT
            if DIFT_action_ID == len(neighor_IDs): #No Trap
                action_set_DIFT[ii] = -1
            else: #Trap when trap is allowed else No Trap as a default action
                if len(self.trap_set[current_state_ID]) == 0:
                    action_set_DIFT[ii] = -1
                else:
                    action_set_DIFT[ii] = neighor_IDs[DIFT_action_ID]

            # Recording the actions chosen by APT
            if APT_action_ID == len(neighor_IDs): #Dropout
                action_set_APT[ii] = -1
            else: #Transitioning to a neighboring state in the state space of DIFT-APT game
                action_set_APT[ii] = neighor_IDs[APT_action_ID]

            # Determining next state and reward of the game based on the next state of the game
            ss_entry_des_ID = 0  # Inidicates whether the current state is a entry point, a destination related state, or a normal state
            num_stages_with_entry = len(self.ss_entry_dest)
            for nn in range(0, num_stages_with_entry):
                entry_or_destination_flag = current_state_ID in self.ss_entry_dest[nn]

                if int(entry_or_destination_flag) == 1:
                    ss_entry_des_ID = nn
                    break
                else:
                    ss_entry_des_ID = -1

            if current_state_ID == self.s0:  # At pseudo node
                neighbor_IDs = [index for index in range(self.N_ss) if self.state_transition_matrix[current_state_ID][index] == 1]
                next_state = neighbor_IDs[APT_action_ID]
            elif ss_entry_des_ID == -1 or ss_entry_des_ID == 0:  # At a state where tagging is allowed (i.e., At a entry point or a normal state)
                neighbor_IDs = [index for index in range(self.N_ss) if self.state_transition_matrix[current_state_ID][index] == 1]
                if len(neighbor_IDs) > DIFT_action_ID:  # Decide to Trap at a neighbor
                    reward_DIFT = self.CD[neighbor_IDs[DIFT_action_ID]]

                if len(neighbor_IDs) > APT_action_ID:  # APT not dropping out (i.e., transitioning to a neighbor)

                    if DIFT_action_ID == APT_action_ID:
                        FN_prob_dist = np.zeros(2)
                        FN_prob_dist[0] = self.FN[current_state_ID]
                        FN_prob_dist[1] = 1 - self.FN[current_state_ID]
                        success = Supp_Func_Obj.act_choice(FN_prob_dist)
                        if success == 1:  # APT Detected
                            next_state = self.s0
                            reward_DIFT = reward_DIFT + self.DIFT_win[self.stage_ID[current_state_ID]]
                            reward_APT = -self.APT_win[self.stage_ID[current_state_ID]]
                        else:  # Miss detection of APT due to false negatives
                            next_state = neighbor_IDs[APT_action_ID]
                            dest_flag = next_state in self.ss_entry_dest[self.stage_ID[current_state_ID] + 1]
                            if int(dest_flag) == 1:  # Destination node
                                reward_DIFT = reward_DIFT + self.DIFT_lose[self.stage_ID[current_state_ID]]
                                reward_APT = self.APT_win[self.stage_ID[current_state_ID]]
                    else:  # Not detected (missed due to false positives)
                        next_state = neighbor_IDs[APT_action_ID]
                        dest_flag = next_state in self.ss_entry_dest[self.stage_ID[current_state_ID] + 1]
                        if int(dest_flag) == 1:  # Destination node
                            reward_DIFT = reward_DIFT + self.DIFT_lose[self.stage_ID[current_state_ID]]
                            reward_APT = self.APT_win[self.stage_ID[current_state_ID]]

                else:  # Dropout
                    next_state = self.s0
                    reward_DIFT = reward_DIFT - self.APT_drop[self.stage_ID[current_state_ID]]
                    reward_APT = self.APT_drop[self.stage_ID[current_state_ID]]

            else:  # Desination related state
                neighor_IDs = [index for index in range(self.N_ss) if self.state_transition_matrix[current_state_ID][index] == 1]
                next_state = neighor_IDs[APT_action_ID]

            current_state = next_state


            Avg_reward_DIFT[ii+1] = (ii*Avg_reward_DIFT[ii] + reward_DIFT)/(ii+1)
            Avg_reward_APT[ii+1] = (ii*Avg_reward_APT[ii] + reward_APT)/(ii+1)

        Output_Data_average_reward.append(Avg_reward_DIFT) #[0] Average reward DIFT
        Output_Data_average_reward.append(Avg_reward_APT) #[1] Average_reward APT

        return Output_Data_average_reward

    def r_PV_eval(self, player_index, player_action_index, current_state):
        Output_Data_r_PV_eval = []
        r_term = 0
        PV_term = 0
        current_state_ID = current_state

        if player_index == 0: #DIFT, Fix DIFT action and average over APT's policy
            for aa in range(0, len(self.Policy_Data[self.t][1][current_state])):
                reward_DIFT = 0
                PV_DIFT = 0

                #Choosing actions correpsonding next grid positions
                #DIFT
                DIFT_action_ID = player_action_index
                #APT
                APT_action_ID = aa

                #Determining next state and reward of the game based on the next state of the game
                ss_entry_des_ID = 0  # Inidicates whether the current state is a entry point, a destination related state, or a normal state
                num_stages_with_entry = len(self.ss_entry_dest)
                for nn in range(0, num_stages_with_entry):
                    entry_or_destination_flag = current_state_ID in self.ss_entry_dest[nn]

                    if int(entry_or_destination_flag) == 1:
                        ss_entry_des_ID = nn
                        break
                    else:
                        ss_entry_des_ID = -1

                if current_state_ID == self.s0:  # At pseudo node
                    neighbor_IDs = [index for index in range(self.N_ss) if self.state_transition_matrix[current_state_ID][index] == 1]
                    next_state = neighbor_IDs[APT_action_ID]
                    PV_DIFT = self.V_Data[self.t][0][next_state]
                elif ss_entry_des_ID == -1 or ss_entry_des_ID == 0:  # At a state where tagging is allowed (i.e., At a entry point or a normal state)
                    neighbor_IDs = [index for index in range(self.N_ss) if self.state_transition_matrix[current_state_ID][index] == 1]
                    if len(neighbor_IDs) > DIFT_action_ID:  # Decide to Trap at a neighbor
                        reward_DIFT = self.CD[neighbor_IDs[DIFT_action_ID]]

                    if len(neighbor_IDs) > APT_action_ID:  # APT not dropping out (i.e., transitioning to a neighbor)

                        if DIFT_action_ID == APT_action_ID:
                            FN_prob_dist = np.zeros(2)
                            FN_prob_dist[0] = self.FN[current_state_ID]
                            FN_prob_dist[1] = 1 - self.FN[current_state_ID]

                            for success in range(0, 2):
                                if success == 1:  # APT Detected
                                    next_state = self.s0
                                    reward_DIFT = reward_DIFT + self.DIFT_win[self.stage_ID[current_state_ID]]*FN_prob_dist[1]
                                    PV_DIFT = PV_DIFT + self.V_Data[self.t][0][next_state]*FN_prob_dist[1]
                                else:  # Miss detection of APT due to false negatives
                                    next_state = neighbor_IDs[APT_action_ID]
                                    dest_flag = next_state in self.ss_entry_dest[self.stage_ID[current_state_ID] + 1]
                                    if int(dest_flag) == 1:  # Destination node
                                        reward_DIFT = reward_DIFT + self.DIFT_lose[self.stage_ID[current_state_ID]]*FN_prob_dist[0]
                                    PV_DIFT = PV_DIFT + self.V_Data[self.t][0][next_state]*FN_prob_dist[0]
                        else:  # Not detected (missed due to false positives)
                            next_state = neighbor_IDs[APT_action_ID]
                            dest_flag = next_state in self.ss_entry_dest[self.stage_ID[current_state_ID] + 1]
                            if int(dest_flag) == 1:  # Destination node
                                reward_DIFT = reward_DIFT + self.DIFT_lose[self.stage_ID[current_state_ID]]
                            PV_DIFT = self.V_Data[self.t][0][next_state]

                    else:  # Dropout
                        next_state = self.s0
                        reward_DIFT = reward_DIFT - self.APT_drop[self.stage_ID[current_state_ID]]
                        PV_DIFT = self.V_Data[self.t][0][next_state]

                else:  # Desination related state
                    neighor_IDs = [index for index in range(self.N_ss) if self.state_transition_matrix[current_state_ID][index] == 1]
                    next_state = neighor_IDs[APT_action_ID]
                    PV_DIFT = self.V_Data[self.t][0][next_state]

                r_term = r_term + reward_DIFT*self.Policy_Data[self.t][1][current_state][aa]
                PV_term = PV_term + PV_DIFT*self.Policy_Data[self.t][1][current_state][aa]
        else: #APT, Fix APT action and average over DIFT's policy
            for dd in range(0,len(self.Policy_Data[self.t][0][current_state])):
                reward_APT = 0
                PV_APT = 0

                # Choosing actions correpsonding next grid positions
                # DIFT
                DIFT_action_ID = dd
                # APT
                APT_action_ID = player_action_index

                # Determining next state and reward of the game based on the next state of the game
                ss_entry_des_ID = 0  # Inidicates whether the current state is a entry point, a destination related state, or a normal state
                num_stages_with_entry = len(self.ss_entry_dest)
                for nn in range(0, num_stages_with_entry):
                    entry_or_destination_flag = current_state_ID in self.ss_entry_dest[nn]

                    if int(entry_or_destination_flag) == 1:
                        ss_entry_des_ID = nn
                        break
                    else:
                        ss_entry_des_ID = -1

                if current_state_ID == self.s0:  # At pseudo node
                    neighbor_IDs = [index for index in range(self.N_ss) if self.state_transition_matrix[current_state_ID][index] == 1]
                    next_state = neighbor_IDs[APT_action_ID]
                    PV_APT = self.V_Data[self.t][1][next_state]
                elif ss_entry_des_ID == -1 or ss_entry_des_ID == 0:  # At a state where tagging is allowed (i.e., At a entry point or a normal state)
                    neighbor_IDs = [index for index in range(self.N_ss) if self.state_transition_matrix[current_state_ID][index] == 1]

                    if len(neighbor_IDs) > APT_action_ID:  # APT not dropping out (i.e., transitioning to a neighbor)

                        if DIFT_action_ID == APT_action_ID:
                            FN_prob_dist = np.zeros(2)
                            FN_prob_dist[0] = self.FN[current_state_ID]
                            FN_prob_dist[1] = 1 - self.FN[current_state_ID]

                            for success in range(0, 2):
                                if success == 1:  # APT Detected
                                    next_state = self.s0
                                    reward_APT = reward_APT - self.APT_win[self.stage_ID[current_state_ID]]*FN_prob_dist[1]
                                    PV_APT = PV_APT + self.V_Data[self.t][1][next_state]*FN_prob_dist[1]
                                else:  # Miss detection of APT due to false negatives
                                    next_state = neighbor_IDs[APT_action_ID]
                                    dest_flag = next_state in self.ss_entry_dest[self.stage_ID[current_state_ID] + 1]
                                    if int(dest_flag) == 1:  # Destination node
                                        reward_APT = reward_APT + self.APT_win[self.stage_ID[current_state_ID]]*FN_prob_dist[0]
                                    PV_APT = PV_APT + self.V_Data[self.t][1][next_state]*FN_prob_dist[0]
                        else:  # Not detected (missed due to false positives)
                            next_state = neighbor_IDs[APT_action_ID]
                            dest_flag = next_state in self.ss_entry_dest[self.stage_ID[current_state_ID] + 1]
                            if int(dest_flag) == 1:  # Destination node
                                reward_APT = self.APT_win[self.stage_ID[current_state_ID]]
                            PV_APT = self.V_Data[self.t][1][next_state]

                    else:  # Dropout
                        next_state = self.s0
                        reward_APT = self.APT_drop[self.stage_ID[current_state_ID]]
                        PV_APT = self.V_Data[self.t][1][next_state]

                else:  # Desination related state
                    neighor_IDs = [index for index in range(self.N_ss) if self.state_transition_matrix[current_state_ID][index] == 1]
                    next_state = neighor_IDs[APT_action_ID]
                    PV_APT = self.V_Data[self.t][1][next_state]

                r_term = r_term + reward_APT*self.Policy_Data[self.t][0][current_state][dd]
                PV_term = PV_term + PV_APT*self.Policy_Data[self.t][0][current_state][dd]

        Output_Data_r_PV_eval.append(r_term)
        Output_Data_r_PV_eval.append(PV_term)
        return Output_Data_r_PV_eval

    def obj_evaluate(self, rho_Data):
        Output_Data_obj_evaluate = []
        obj_value = 0
        obj_DIFT = 0
        obj_APT = 0

        #Evaluating obj. function corresponding to DIFT
        for ii in range(0, self.N_ss):
            r_term = 0
            PV_term = 0

            for dd in range(0, len(self.Policy_Data[self.t][0][ii])):
                player_index = 0
                player_action_index = dd
                current_state = ii
                Output_Data_r_PV_eval = self.r_PV_eval(player_index, player_action_index, current_state)
                r_tmp = Output_Data_r_PV_eval[0]
                PV_tmp = Output_Data_r_PV_eval[1]
                r_term = r_term + r_tmp*self.Policy_Data[self.t][0][ii][dd]
                PV_term = PV_term + PV_tmp*self.Policy_Data[self.t][0][ii][dd]

            obj_DIFT = obj_DIFT + rho_Data[0] + self.V_Data[self.t][0][ii] - r_term - PV_term


        #Evaluating obj. function corresponding to adversary
        for ii in range(0, self.N_ss):
            r_term = 0
            PV_term = 0

            for aa in range(0, len(self.Policy_Data[self.t][1][ii])):
                player_index = 1
                player_action_index = aa
                current_state = ii
                Output_Data_r_PV_eval = self.r_PV_eval(player_index, player_action_index, current_state)
                r_tmp = Output_Data_r_PV_eval[0]
                PV_tmp = Output_Data_r_PV_eval[1]
                r_term = r_term + r_tmp*self.Policy_Data[self.t][1][ii][aa]
                PV_term = PV_term + PV_tmp*self.Policy_Data[self.t][1][ii][aa]

            obj_APT = obj_APT + rho_Data[1] + self.V_Data[self.t][1][ii] - r_term - PV_term

        obj_value = obj_DIFT + obj_APT

        Output_Data_obj_evaluate.append(obj_value)
        Output_Data_obj_evaluate.append(obj_DIFT)
        Output_Data_obj_evaluate.append(obj_APT)

        return Output_Data_obj_evaluate


