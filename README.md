# MA-ARNE-Algorithm-for-DIFT-APT-games

SAMPLE CODE OF MA-ARNE ALGORITHM FOR AVERAGE REWARD Dynamic Information Flow Tracking-Advanced Persistent Threats(DIFT-APT) GAMES

(*) For detailed explanation of the MA-ARNE ALGORITHM and DIFT-APT GAME MODEL please refer to:
   - paper: "A Multi-Agent Reinforcement Learning Approach for Dynamic Information Flow Tracking Games for Advanced Persistent Threats"
   - website: https://adapt.ece.uw.edu/
   
(*) You may freely redistribute and use this sample code, with or without modification, provided you include the original copyright notice and use restrictions.

Disclaimer: THE SAMPLE CODE IS PROVIDED "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A 
PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL DINUKA OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL 
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) SUSTAINED BY YOU 
OR A THIRD PARTY, HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT ARISING IN ANY WAY OUT OF THE USE OF THIS SAMPLE 
CODE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

For additional information, contact:
Dinuka Sahabandu
email: sdinuka@uw.edu

Following files are needed for this code to run:

(1) DIFT_APT_Game.m :: This is the main file you need to run for the simulations

(2) MA_ARNE.m :: MA-ARNE algorithm implementation for DIFT-APT games

(3) act_choice.m :: Function that choose actions for the players DIFT and APT probabilistically in DIFT-APT game

(4) find_average_reward.m :: Function that computes avergae reward for the players for a given policy

(5) obj_evaluate.m :: Function evaluates objective functions (refer to the paper "A Multi-Agent Reinforcement Learning Approach for Dynamic Information Flow Tracking Games for Advanced Persistent Threats" for formal definitions)

(6) policy_projection.m :: Function projects the policies to probability simplex in MA-ARNE algorithm 

(7) r_PV_eval.m :: Function computes variable values in obj_evaluate.m

(8) random_initial_state.m :: Function chooses a random initial state for the DIFT-APT game to improve exploration in MA_ARNE.m

(9) Experiment_Plots.m :: Function evaluates the performance of algorithm and output three figures (in Figure 3: [Case 1] - ARNE policy, [Case 2]- Uniform policy, [Case 3]- Cut policy

(10) Ransomeware_Data.mat :: Data file containing state space of DIFT-APT game extracted from Ransomware system logs

Acknowledgement: This work was supported by ONR grant N00014-16-1-2710 P00002 and DARPA TC grant DARPA FA8650-15-C-7556.
