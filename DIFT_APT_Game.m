%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                   SAMPLE CODE OF MA-ARNE ALGORITHM FOR AVERAGE REWARD Dynamic Information Flow Tracking-Advanced Persistent Threats(DIFT-APT) GAMES
%
%(*) For detailed explanation of the MA-ARNE ALGORITHM and DIFT-APT GAME MODEL please refer to:
%    - paper: "A Multi-Agent Reinforcement Learning Approach for Dynamic Information Flow Tracking Games for Advanced Persistent Threats"
%    - website: https://adapt.ece.uw.edu/
%(*) You may freely redistribute and use this sample code, with or without modification, provided you include the original copyright notice and use restrictions.
% 
%Disclaimer: THE SAMPLE CODE IS PROVIDED "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A 
%PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL DINUKA OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL 
%DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) SUSTAINED BY YOU 
%OR A THIRD PARTY, HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT ARISING IN ANY WAY OUT OF THE USE OF THIS SAMPLE 
%CODE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
% 
% For additional information, contact:
% Dinuka Sahabandu
% email: sdinuka@uw.edu
%
% Following files are needed for this code to run:
%
% DIFT_APT_Game.m :: This is the main file you need to run for the simulations
% MA_ARNE.m :: MA-ARNE algorithm implementation for DIFT-APT gamez
% act_choice.m :: Function that choose actions for the players DIFT and APT probabilistically in DIFT-APT game
% find_average_reward.m :: Function that computes avergae reward for the players for a given policy
% obj_evaluate.m :: Function evaluates objective functions (refer to the paper "A Multi-Agent Reinforcement Learning Approach for Dynamic Information Flow Tracking Games for Advanced Persistent Threats" for formal definitions)
% policy_projection.m :: Function projects the policies to probability simplex in MA-ARNE algorithm 
% r_PV_eval.m :: Function computes variable values in obj_evaluate.m
% random_initial_state.m :: Function chooses a random initial state for the DIFT-APT game to improve exploration in MA_ARNE.m
% Experiment_Plots.m :: Function evaluates the performance of algorithm and
% output three figures (in Figure 3: [Case 1] - ARNE policy, [Case 2]- Uniform policy, [Case 3]- Cut policy
% Ransomeware_Data.mat :: Data file containing state space of DIFT-APT game extracted from Ransomware system logs
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clc
clear all
close all


%Load RAIN dataset
load('Ransomeware_Data.mat','state_transition_matrix','node_names')
%state_transition_matrix: A transition matrix corresponding to DIFT-APT game's state space (includs virtual state s_0)

N_ss = length(state_transition_matrix); %Total number of states in DIFT-APT game corresponsding to ransomware attack

%Define the entry points an destination nodes of the attack here
ss_entry_dest{1} = [1 7]; %Entry points state ID
ss_entry_dest{2} = [3]; %stage one destinations state ID
ss_entry_dest{3} = [10];%stage two destinations state ID
ss_entry_dest{4} = [17]; %final destination state ID
s_0 = 21; %Virtual state ID 

%Define Rewards and Penalties here
APT_drop = [-30 -50 -70]; %Penalties for APT quitting the attack corresponding to each stage (Rewards for DIFT in this cases are set to -APT_drop)
APT_win = [20 40 60]; %Reward for APT reaching corresponding destinations in each attack stage (Penalty for APT getting detected in these casea are set to -APT win)
DIFT_win = [40 80 120]; %Reward to DIFT detecting APT in each attack stage 
DIFT_lose = [-30 -60 -90]; %Penalty for DIFT when APT reach a corresponding destinations in each attack stage 

episodes_sim = 5000; %Totla number of iterations 


[t, Policy_Data, V_Data, stage_ID, FN, CD, trap_set] = MA_ARNE(state_transition_matrix, node_names, ss_entry_dest,N_ss,s_0,episodes_sim, APT_drop, APT_win, DIFT_win, DIFT_lose);
Experiment_Plots(t, Policy_Data, V_Data, N_ss, stage_ID, ss_entry_dest, FN, CD, APT_drop, APT_win, DIFT_win, DIFT_lose,state_transition_matrix, trap_set)