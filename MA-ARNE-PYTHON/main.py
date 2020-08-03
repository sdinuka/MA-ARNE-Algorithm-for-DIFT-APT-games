from scipy.io import loadmat
from Experiment_plots import *
from MA_ARNE import *

N_ss = 21
ss_entry_dest = []
ss_entry_dest.append([0, 6])
ss_entry_dest.append([2])
ss_entry_dest.append([9])
ss_entry_dest.append([16])
s_0 = 20

#Load state space of the DOFT-APT game as a transition matrix
IFG_Data = loadmat('Ransomeware_Data.mat')
state_transition_matrix = IFG_Data['state_transition_matrix']
node_names = IFG_Data['node_names']
stage_indicators = ['v1', 'v2', 'v3']
#Define DIFT-APT game parameters here (e.g., rewards, penalties, costs and false negatives)
episodes_sim = int(0.5e5)
APT_drop = [-30, -50, -70]
APT_win = [20, 40, 60]
DIFT_win = [20, 40, 60]
DIFT_lose = [-20, -40, -60]
FN_stages = [0.2, 0.1, 0.05]
CD_stages = [-3, -2, -1]


MA_ARNE_Obj = MA_ARNE_Algo(state_transition_matrix, node_names, ss_entry_dest, N_ss, s_0, episodes_sim, APT_drop, APT_win, DIFT_win, DIFT_lose, FN_stages, CD_stages, stage_indicators)
MA_ARNE_Output = MA_ARNE_Obj.MA_ARNE_Algorithm()

t_sim = MA_ARNE_Output[0] #Termination time of the MA-ARNE algorithm
Policy_Data = MA_ARNE_Output[1] #Policy Data
V_Data = MA_ARNE_Output[2] #Value Data
stage_ID = MA_ARNE_Output[3] #stage_ID
FN = MA_ARNE_Output[4] #FN
CD = MA_ARNE_Output[5] #CD
trap_set = MA_ARNE_Output[6] #trap_set

Experiment_plots_obj = MA_ARNE_plots(t_sim, Policy_Data, V_Data, N_ss, stage_ID, ss_entry_dest, FN, CD, APT_drop, APT_win, DIFT_win,DIFT_lose,state_transition_matrix, trap_set, s_0)
Experiment_plots_obj.convergence_plots()


