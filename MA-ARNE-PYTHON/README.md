<b><u><i>Summary</i></u></b>

Dynamic Information Flow Tracking (DIFT) is a defense mechanism that dynamically track the usage of information flows in a computer system during program executions. Advanced Persistent Threats (APTs) are sophisticated, stealthy, long-term cyberattacks that target specific systems. Although DIFT has been used for detecting APTs, wide range security analysis using DIFT results in a significant increase in performance overhead and high rates of false-positives and false-negatives. This code presents a game-theoretic implementation of the strategic interaction between APT and DIFT. The DIFT-APT game is a nonzero-sum stochastic game with imperfect information and average reward structure. The average payoff structure captures the long-term behavior of the APT’s interactions with the victim system.  Additionally, the game has incomplete information structure as the transition probabilities (false-positive and false-negative rates) are unknown. In [1], we showed that the state space of the game has a unichain structure. Utilizing the unichain structure we proposed Multi-Agent Average Reward Nash Equilibrium (MA-ARNE) algorithm to compute an average reward Nash equilibrium of the game and proved convergence in [1].

<b><u><i>Code description</i></u></b>

This code presents the Python (version 3.7.7) implementation of the MA-ARNE algorithm. The MA-ARNE algorithm is a multiple-time scale stochastic approximation algorithm that learns an equilibrium solution of the DIFT-APT game. MA-ARNE python code contains four classes of functions, Supplementary_Functions, MA-ARNE, Evaluation_Functions and Experiment_plots. Detailed description of the python code is given below. 

•	<b>GitHub link:</b> https://github.com/sdinuka/MA-ARNE-Algorithm-for-DIFT-APT-games/blob/master/MA-ARNE-PYTHON

•	<b>Python version -</b> 3.7.7

•	<b>Python libraries required -</b> numpy, scipy.io, copy, statistics, matplotlib.pylab

•	<b>Function description</b> 

•	<b>main.py:</b> Loads the state space of DIFT-APT game (as a .mat file containing a transition matrix) and defines the reward, penalty and cost 		parameters of DIFT-APT game. Execution of "main.py" computes the Nash equilibrium policies in DIFT-APT game using MA-ARNE algorithm and provides 		convergence plots of value functions, average rewards and objective functions. 

•	<b>Class: Supplementary_Functions.py</b>
        This class implements the following three functions
	     
•	<b>act_choice -</b> Function that chooses actions for the players DIFT and APT probabilistically (for a given policy of DIFT/APT) in DIFT-APT game

•	<b>policy_projection -</b> Function projects the updated policies of DIFT and APT in MA-ARNE algorithm to probability simplex 

•	<b>random_initial_state -</b> Function chooses a random initial state for the DIFT-APT game to improve exploration in MA_ARNE algorithm 

•	<b>Class: MA_ARNE.py</b> 
        This class implements the multiple-time scale stochastic approximation algorithm, MA-ARNE, that learns an equilibrium solution of the DIFT-APT game.
	
•	<b>Class: Evaluation_Functions.py</b> 
	This class consists of the following three functions
	
•	<b>find_average_reward -</b> Function that computes average reward of DIFT and APT corresponding to a given policy pair of DIFT and APT

•	<b>obj_evaluate -</b> Function evaluates objective functions of DIFT and APT (please refer to [1] for the definition of the objective functions of DIFT 	and APT)

•	<b>r_PV_eval -</b> Function computes variable values (immediate reward at time t and expected reward at time t+1) required to calculate the objective 		function values of DIFT and APT in function "obj_evaluate" 

•	<b>Class: Experiment_plots.py</b>
	This class includes the function "convergence_plots" which plots the 	convergence of value functions, average reward values, objective function 		values of DIFT and APT, and the total objective function value.
	
The experiments in this code used a ransomware attack dataset collected using Refinable Attack INvestigation (RAIN) framework [2]. For more details on the algorithm and results see [1].

<b>Note:</b> You may freely redistribute and use this sample code, with or without modification, provided you include the original Copyright notice and use restrictions.

<b>Disclaimer:</b> THE SAMPLE CODE IS PROVIDED "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL DINUKA SAHABANDU OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) SUSTAINED BY YOU OR A THIRD PARTY, HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT ARISING IN ANY WAY OUT OF THE USE OF THIS SAMPLE CODE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

<b><u><i>Related papers</b></u></i>
	
[1] Dinuka Sahabandu, Shana Moothedath,  Joey Allen, Linda Bushnell, Wenke Lee, and Radha Poovendran, “A Multi-Agent Reinforcement Learning Approach for Dynamic Information Flow Tracking Games for Advanced Persistent Threats”. 

ArXiv link: https://arxiv.org/pdf/2007.00076.pdf

Website: https://adapt.ece.uw.edu/

[2] Yang Ji, Sangho Lee, Evan Downing, Weiren Wang, Mattia Fazzini, Taesoo Kim, Alessandro Orso, and Wenke Lee. "Rain: Refinable attack investigation with on-demand inter-process information flow tracking." In Proceedings of the 2017 ACM SIGSAC Conference on Computer and Communications Security, pp. 377-390. 2017.

For additional information, contact: Dinuka Sahabandu, email: sdinuka@uw.edu

<b>Acknowledgement:</b> This work was supported by ONR grant N00014-16-1-2710 P00002, DARPA TC grant DARPA FA8650-15-C-7556, and ARO grant W911NF-16-1-0485.






  

