from Supplementary_Functions import *
import copy

class MA_ARNE_Algo:
    state_transition_matrix = None
    node_names = None
    ss_entry_dest = None
    N_ss = None
    s_0 = None
    episodes_sim = None
    APT_drop = None
    APT_win = None
    DIFT_win = None
    DIFT_lose = None
    FN_stages = None
    CD_stages = None
    stage_indicators = None
    V_Data = None

    def __init__(self, state_transition_matrix, node_names, ss_entry_dest, N_ss, s_0, episodes_sim, APT_drop, APT_win, DIFT_win, DIFT_lose, FN_stages, CD_stages, stage_indicators):
        self.state_transition_matrix = state_transition_matrix
        self.node_names = node_names
        self.ss_entry_dest = ss_entry_dest
        self.N_ss = N_ss
        self.s_0 = s_0
        self.episodes_sim = episodes_sim
        self.APT_drop = APT_drop
        self.APT_win = APT_win
        self.DIFT_win = DIFT_win
        self.DIFT_lose = DIFT_lose
        self.FN_stages = FN_stages
        self.CD_stages = CD_stages
        self.stage_indicators = stage_indicators


    def MA_ARNE_Algorithm(self):

        MA_ARNE_Output = []
        Supp_Func_Obj = Supp_functions(self.N_ss, self.ss_entry_dest)

        DIFT_avoid = []
        APT_avoid = []
        num_stages_with_entry = len(self.ss_entry_dest)

        for ii in range(0,num_stages_with_entry):
            if ii > 0: #Beyond the entry points of the attack (i.e., stage 1, stage 2, ..., destinations)
                for add_element in self.ss_entry_dest[ii]:
                    DIFT_avoid.append(add_element)
                    APT_avoid.append(add_element)

        DIFT_avoid.append(self.s_0) #No security check allowed at the virtual state (vitual state models the Initial Recon stage of the attack

        #Find at each state what neighboring nodes allow security checks for DIFT excluding the DIFT_avoid states in the neighbor set
        trap_set = []
        for ii in range(0, self.N_ss):
            allowed_flag = ii in DIFT_avoid
            if int(allowed_flag) == 0:
                trap_set_tmp = [index for index in range(self.N_ss) if self.state_transition_matrix[ii][index] == 1]
                trap_set.append(trap_set_tmp)
            else:
                trap_set.append([])

        FN = []
        CD = []
        stage_ID = []
        num_stages = num_stages_with_entry - 1
        stage_numbers = list(range(0, num_stages))

        for ii in range(0, self.N_ss):
            node_tag = self.node_names[0][ii][0][-2:]
            for jj in range(0, num_stages):
                if node_tag == self.stage_indicators[jj]:
                    FN.append(self.FN_stages[jj])
                    CD.append(self.CD_stages[jj])
                    stage_ID.append(stage_numbers[jj])
                elif node_tag == '_0':
                    FN.append(0)
                    CD.append(0)
                    stage_ID.append(-1)

        #Initialization of values (V_Data) and average reward values (rho_Data)
        V_Data = []
        rho_Data = []

        V_Data.append(np.zeros([2, self.N_ss])) #{t}(DIFT:APT,state-1:state-N_ss)
        rho_Data.append(np.zeros([2, 1])) #{t}(DIFT:APT)

        #Initializing player policies and gradient data
        Policy_Data = [] #{t}{player}{state}(action-1-player:action-end-player)
        Gradient_Data = [] #{t}{player}{state}(action-1-player:action-end-player)
        Policy_Data_state_DIFT = []
        Policy_Data_state_APT = []
        Gradient_Data_state_DIFT = []
        Gradient_Data_state_APT = []

        for ii in range(0,self.N_ss):
            #Initializing player one policy
            num_act_DIFT = len(trap_set[ii]) + 1 #+1 for no trapping when trapping is allowed, when trapping is not allowed this represnts a pseduo action with probability one

            Policy_Data_state_DIFT.append(np.ones([num_act_DIFT])/num_act_DIFT)
            Gradient_Data_state_DIFT.append(np.zeros([num_act_DIFT])/num_act_DIFT)

            #Initializing player two policy
            neighbor_states = [neighbor_index for neighbor_index in range(self.N_ss) if self.state_transition_matrix[ii][neighbor_index] == 1]
            APT_avoid_flag = ii in APT_avoid

            if int(APT_avoid_flag) == 0 and ii != self.s_0: #States where droppingout is allowed
                num_act_APT = len(neighbor_states) + 1 # +1 for dropout
            else:
                num_act_APT = len(neighbor_states)

            Policy_Data_state_APT.append(np.ones([num_act_APT]) / num_act_APT)
            Gradient_Data_state_APT.append(np.zeros([num_act_APT]) / num_act_APT)


        Policy_Data_tmp = []
        Gradient_Data_tmp = []

        Policy_Data_tmp.append(Policy_Data_state_DIFT)
        Policy_Data_tmp.append(Policy_Data_state_APT)

        Gradient_Data_tmp.append(Gradient_Data_state_DIFT)
        Gradient_Data_tmp.append(Gradient_Data_state_APT)

        Policy_Data.append(Policy_Data_tmp)
        Gradient_Data.append(Gradient_Data_tmp)

        #Setting the initial state of the game to s0
        initial_state = self.s_0
        current_state = initial_state

        #Initializing variables that keep track of number of times each state, (state, DIFT_action), (state, APT_action) and (state, DIFT_action, APT_action) has been used
        State_Tracker = np.zeros(self.N_ss)
        State_Action_Tracker = []
        State_BothActions_Tracker = []

        for ii in range(0, self.N_ss):
            num_of_acts_DIFT = len(Policy_Data[0][0][ii])
            num_of_acts_APT = len(Policy_Data[0][1][ii])

            State_Action_Tracker_tmp  = []
            State_Action_Tracker_tmp.append(np.zeros(num_of_acts_DIFT))
            State_Action_Tracker_tmp.append(np.zeros(num_of_acts_APT))
            State_Action_Tracker.append(State_Action_Tracker_tmp)

            State_BothActions_Tracker.append(np.zeros((num_of_acts_DIFT, num_of_acts_APT)))

        t = 0
        s_0_counter = 1 #For counting number of times s_0 visited
        counter = 1 #For counting number of iterations after 7000th iteration
        eta = 0.99 #Set a value close to 1 for fast convergence while maintaining the values of the solution (e.g., value functions, average reward, player policies) closer to the optimal
        for episode in range(0, self.episodes_sim):
            reward_DIFT = 0 #DIFT's reward at the current episode
            reward_APT = 0 #APTs reward at the current episode
            #Finding the state ID
            current_state_ID = current_state

            # Choosing actions for DIFT and APT based on their respective policies at the current iteration
            # DIFT
            #print(current_state_ID,Policy_Data[t][0])
            DIFT_action_ID = Supp_Func_Obj.act_choice(Policy_Data[t][0][current_state_ID])
            # APT
            APT_action_ID = Supp_Func_Obj.act_choice(Policy_Data[t][1][current_state_ID])

            #Determining next state and reward of the game based on the next state of the game
            ss_entry_des_ID = 0 #Inidicates whether the current state is a entry point, a destination related state, or a normal state
            for nn in range(0, num_stages_with_entry):
                entry_or_destination_flag = current_state_ID in self.ss_entry_dest[nn]

                if int(entry_or_destination_flag) == 1:
                    ss_entry_des_ID = nn
                    break
                else:
                    ss_entry_des_ID = -1

            if current_state_ID == self.s_0:  #At pseudo node
               neighbor_IDs = [index for index in range(self.N_ss) if self.state_transition_matrix[current_state_ID][index] == 1]
               next_state  = neighbor_IDs[APT_action_ID]
            elif ss_entry_des_ID == -1 or ss_entry_des_ID == 0: #At a state where tagging is allowed (i.e., At a entry point or a normal state)
                neighbor_IDs = [index for index in range(self.N_ss) if self.state_transition_matrix[current_state_ID][index] == 1]
                if len(neighbor_IDs) > DIFT_action_ID: #Decide to Trap at a neighbor
                    reward_DIFT = CD[neighbor_IDs[DIFT_action_ID]]

                if len(neighbor_IDs) > APT_action_ID: #APT not dropping out (i.e., transitioning to a neighbor)

                    if DIFT_action_ID == APT_action_ID:
                        FN_prob_dist = np.zeros(2)
                        FN_prob_dist[0] = FN[current_state_ID]
                        FN_prob_dist[1] = 1 - FN[current_state_ID]
                        success = Supp_Func_Obj.act_choice(FN_prob_dist)
                        if success == 1: # APT Detected
                            next_state = self.s_0
                            reward_DIFT = reward_DIFT + self.DIFT_win[stage_ID[current_state_ID]]
                            reward_APT = -self.APT_win[stage_ID[current_state_ID]]
                        else: #Miss detection of APT due to false negatives
                            next_state = neighbor_IDs[APT_action_ID]
                            dest_flag = next_state in self.ss_entry_dest[stage_ID[current_state_ID]+1]
                            if int(dest_flag) == 1: #Destination node
                                reward_DIFT = reward_DIFT + self.DIFT_lose[stage_ID[current_state_ID]]
                                reward_APT = self.APT_win[stage_ID[current_state_ID]]
                    else: #Not detected (missed due to false positives)
                        next_state = neighbor_IDs[APT_action_ID]
                        dest_flag = next_state in self.ss_entry_dest[stage_ID[current_state_ID] + 1]
                        if int(dest_flag) == 1: #Destination node
                            reward_DIFT = reward_DIFT + self.DIFT_lose[stage_ID[current_state_ID]]
                            reward_APT = self.APT_win[stage_ID[current_state_ID]]

                else: #Dropout
                    next_state = self.s_0
                    reward_DIFT = reward_DIFT - self.APT_drop[stage_ID[current_state_ID]]
                    reward_APT = self.APT_drop[stage_ID[current_state_ID]]

            else: #Desination related state
                neighor_IDs = [index for index in range(self.N_ss) if self.state_transition_matrix[current_state_ID][index] == 1]
                next_state = neighor_IDs[APT_action_ID]

            next_state_ID = next_state

            ##### Tracking the number of times state, (state, DIFT_action), (state, APT_action)

            State_Tracker[current_state_ID] = State_Tracker[current_state_ID] + 1
            State_Action_Tracker[current_state_ID][0][DIFT_action_ID] = State_Action_Tracker[current_state_ID][0][DIFT_action_ID] + 1
            State_Action_Tracker[current_state_ID][1][APT_action_ID] = State_Action_Tracker[current_state_ID][1][APT_action_ID] + 1
            State_BothActions_Tracker[current_state_ID][DIFT_action_ID][APT_action_ID] = State_BothActions_Tracker[current_state_ID][DIFT_action_ID][APT_action_ID] + 1

            ##### MA-ARNE Algorithm

            ######## Defining learning rates

            if episode < 7000:
                gamma_1 = 0.1 #learning rate for average reward
                gamma_2 = 0.8 #learning rate for V updates and Gradient updates
                gamma_3 = 0.01 #learning rate policy updates
            else:
                gamma_1 = 1/(1 + counter*np.log(counter)) #Learning rate for average reward
                gamma_2 = 1/(State_Tracker[current_state_ID]) #Learning rate for V updates and Gradient updates
                gamma_3 = 1/counter #learning rate for policy updates
                counter = counter + 1

            ###########   Update Equations ################

            ######### Average Reward Updates
            rho_Data_tmp = copy.deepcopy(rho_Data[t])
            rho_Data_tmp[0] = (1-gamma_1)*rho_Data[t][0] + gamma_1*((t*rho_Data[t][0] + reward_DIFT)/(t+1))
            rho_Data_tmp[1] = (1-gamma_1)*rho_Data[t][1] + gamma_1*((t*rho_Data[t][1] + reward_APT)/(t+1))
            rho_Data.append(rho_Data_tmp)

            ######### Update value functions(V)
            V_Data_tmp = copy.deepcopy(V_Data[t])
            V_Data_tmp[0][current_state_ID] = (1-gamma_2)*V_Data[t][0][current_state_ID] + gamma_2*(reward_DIFT - rho_Data[t+1][0] + eta*V_Data[t][0][next_state_ID])
            V_Data_tmp[1][current_state_ID] = (1-gamma_2)*V_Data[t][1][current_state_ID] + gamma_2*(reward_APT - rho_Data[t+1][1] + eta*V_Data[t][1][next_state_ID])
            V_Data.append(V_Data_tmp)

            ######## Gradient Updates
            V_term_DIFT = reward_DIFT - rho_Data[t+1][0] + V_Data[t][0][next_state_ID] - V_Data[t][0][current_state_ID]
            V_term_APT = reward_APT - rho_Data[t+1][1] + V_Data[t][1][next_state_ID] - V_Data[t][1][current_state_ID]

            Gradient_Data_tmp = copy.deepcopy(Gradient_Data[t])
            Gradient_Data_tmp[0][current_state_ID][DIFT_action_ID] = (1-gamma_2)*Gradient_Data[t][0][current_state_ID][DIFT_action_ID] + gamma_2*(V_term_DIFT+V_term_APT)
            Gradient_Data_tmp[1][current_state_ID][APT_action_ID] = (1-gamma_2)*Gradient_Data[t][1][current_state_ID][APT_action_ID] + gamma_2*(V_term_DIFT+V_term_APT)
            Gradient_Data.append(Gradient_Data_tmp)

            ########## Policy updates
            Policy_Data_tmp = copy.deepcopy(Policy_Data[t])
            Policy_Data_tmp[0][current_state_ID][DIFT_action_ID] = Policy_Data[t][0][current_state_ID][DIFT_action_ID] - gamma_3*np.sqrt(Policy_Data[t][0][current_state_ID][DIFT_action_ID])*np.abs(V_term_DIFT)*np.sign(-1*Gradient_Data[t+1][0][current_state_ID][DIFT_action_ID])
            Policy_Data_tmp[1][current_state_ID][APT_action_ID] = Policy_Data[t][1][current_state_ID][APT_action_ID] - gamma_3*np.sqrt(Policy_Data[t][1][current_state_ID][APT_action_ID]) * np.abs(V_term_APT)*np.sign(-1*Gradient_Data[t+1][1][current_state_ID][APT_action_ID])
            Policy_Data.append(Policy_Data_tmp)

            ######### Projecting policies to probability simplex
            Policy_Data[t+1][0][current_state_ID] = Supp_Func_Obj.policy_projection(Policy_Data[t+1][0][current_state_ID])
            Policy_Data[t+1][1][current_state_ID] = Supp_Func_Obj.policy_projection(Policy_Data[t+1][1][current_state_ID])

            ######### To improve the exploration in the algorithm: Add a decaying noise to policies of both DIFT and APT when each of them converge to a deterministic policy

            deterministic_entry_DIFT = [index for index in range(len(Policy_Data[t+1][0][current_state_ID])) if Policy_Data[t+1][0][current_state_ID][index] >= 0.95]
            deterministic_entry_APT = [index for index in range(len(Policy_Data[t+1][1][current_state_ID])) if Policy_Data[t+1][1][current_state_ID][index] >= 0.95]

            if episode > 10000:
                noise = 0.01
            elif episode > 5000:
                noise = 0.1
            elif episode > 2500:
                noise = 0.2
            else:
                noise = 0.3

            if len(deterministic_entry_DIFT) == 1:
                for ee in range(0, len(Policy_Data[t+1][0][current_state_ID])):
                    if ee != deterministic_entry_DIFT[0]:
                        Policy_Data[t+1][0][current_state_ID][ee] = Policy_Data[t+1][0][current_state_ID][ee] + noise

                Policy_Data[t+1][0][current_state_ID] = Supp_Func_Obj.policy_projection(Policy_Data[t+1][0][current_state_ID])

            if len(deterministic_entry_APT) == 1:
                for ee in range(0, len(Policy_Data[t+1][1][current_state_ID])):
                    if ee != deterministic_entry_APT[0]:
                        Policy_Data[t+1][1][current_state_ID][ee] = Policy_Data[t+1][1][current_state_ID][ee] + noise

                Policy_Data[t+1][1][current_state_ID] = Supp_Func_Obj.policy_projection(Policy_Data[t+1][1][current_state_ID])

            if next_state == self.s_0: #Counting number of times s_0 is reached as a next state (one round of game is done: i.e., APT dropped out, APT reach the final goal(destination), or DIFT detects APT)
                s_0_counter = s_0_counter + 1

            if np.mod(s_0_counter, 100) == 0: #Ranomly intialize the game to improve the exploration (every 100th time s_0 is visited)
                next_state = Supp_Func_Obj.random_initial_state(State_Tracker)
                s_0_counter = s_0_counter + 1

            current_state = next_state
            t = t + 1
            print('Episode:', episode, 'Complete:', (episode/self.episodes_sim)*100)

        MA_ARNE_Output.append(t) #Termination time of the MA-ARNE algorithm [0]
        MA_ARNE_Output.append(Policy_Data) #Policy Data [1]
        MA_ARNE_Output.append(V_Data) #Value Data [2]
        MA_ARNE_Output.append(stage_ID) #stage_ID [3]
        MA_ARNE_Output.append(FN) #FN [4]
        MA_ARNE_Output.append(CD) #CD [5]
        MA_ARNE_Output.append(trap_set) #trap_set [6]
        print(State_Tracker)

        return MA_ARNE_Output
