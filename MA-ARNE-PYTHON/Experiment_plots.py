from matplotlib.pylab import plt #load plot library
from Evaluation_Functions import *
from statistics import mean

class MA_ARNE_plots:
    t_sim = None
    Policy_Data = None
    V_Data = None
    N_ss = None
    stage_ID = None
    ss_entry_dest = None
    FN = None
    CD = None
    APT_drop = None
    APT_win = None
    DIFT_win = None
    DIFT_lose = None
    state_transition_matrix = None
    trap_set = None
    s0 = None

    def __init__(self, t_sim, Policy_Data, V_Data, N_ss, stage_ID, ss_entry_dest, FN, CD, APT_drop, APT_win, DIFT_win,DIFT_lose,state_transition_matrix, trap_set, s0):
        self.t_sim = t_sim
        self.Policy_Data = Policy_Data
        self.V_Data = V_Data
        self.N_ss = N_ss
        self.stage_ID = stage_ID
        self.ss_entry_dest = ss_entry_dest
        self.FN = FN
        self.CD = CD
        self.APT_drop = APT_drop
        self.APT_win = APT_win
        self.DIFT_win = DIFT_win
        self.DIFT_lose = DIFT_lose
        self.state_transition_matrix = state_transition_matrix
        self.trap_set = trap_set
        self.s0 = s0


    def convergence_plots(self):
        obj_value_plot = []
        obj_value_DIFT_plot = []
        obj_value_APT_plot = []
        iteration_number = []

        t_sim_avg_reward = 1000
        avergae_value_DIFT = []
        avergae_value_APT = []
        average_reward_DIFT = []
        average_reward_APT = []

        for tt in range(0, self.t_sim, 500):
            Eval_Func_Obj = Eval_and_Plots(self.Policy_Data, tt, self.N_ss, self.stage_ID, self.ss_entry_dest, self.CD, self.APT_win, self.APT_drop, self.DIFT_win, self.DIFT_lose, self.trap_set, self.state_transition_matrix, self.FN, self.s0, self.V_Data)
            Output_Data_average_reward = Eval_Func_Obj.find_average_reward(t_sim_avg_reward)
            rho_Data = []

            rho_Data.append(Output_Data_average_reward[0][t_sim_avg_reward-1])
            rho_Data.append(Output_Data_average_reward[1][t_sim_avg_reward-1])

            average_reward_DIFT.append(Output_Data_average_reward[0][t_sim_avg_reward-1])
            average_reward_APT.append(Output_Data_average_reward[1][t_sim_avg_reward-1])
            #Caculating the avrage value
            avergae_value_DIFT.append(mean(self.V_Data[tt][0]))
            avergae_value_APT.append(mean(self.V_Data[tt][1]))

            Output_Data_obj_evaluate = Eval_Func_Obj.obj_evaluate(rho_Data)
            obj_value_plot.append(Output_Data_obj_evaluate[0])
            obj_value_DIFT_plot.append(Output_Data_obj_evaluate[1])
            obj_value_APT_plot.append(Output_Data_obj_evaluate[2])
            iteration_number.append(tt)

        plt.plot(avergae_value_DIFT)
        plt.plot(avergae_value_APT)
        plt.show()

        plt.plot(average_reward_DIFT)
        plt.plot(average_reward_APT)
        plt.show()

        plt.plot(obj_value_plot, label="Obj")
        plt.plot(obj_value_DIFT_plot, label="DIFT Obj")
        plt.plot(obj_value_APT_plot, label="APT Obj")
        plt.legend(loc='lower left')
        plt.show()





