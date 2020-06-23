function [t, Policy_Data, V_Data, stage_ID, FN, CD, trap_set]  = MA_ARNE(state_transition_matrix,node_names,ss_entry_dest,N_ss,s_0,episodes_sim,APT_drop,APT_win,DIFT_win,DIFT_lose)


DIFT_avoid = [];
APT_avoid = [];
for ii = 1:length(ss_entry_dest)
    if ii > 1
        DIFT_avoid = [DIFT_avoid ss_entry_dest{ii}];
        APT_avoid = [APT_avoid ss_entry_dest{ii}];
    end
end
DIFT_avoid = [DIFT_avoid s_0]; %No security check allowed at these locations

%Find at each state what neighboring nodes allow security checks for DIFT
%Excluding the DIFT_avoid states in the neighbor set
trap_set = cell(1);
for ii = 1:N_ss
    allowed_flag = find(DIFT_avoid == ii);
    if isempty(allowed_flag) == 1
        trap_set{ii} = find(state_transition_matrix(ii,:) == 1);
    else
        trap_set{ii} = [];
    end
end

stage_indicators = {'v1','v2','v3'}; %Stages of the attack
stage_numbers = [1 2 3];
FN_stages = [0.2 0.1 0.05];
CD_stages = [-3 -2 -1];
FN = zeros(1,N_ss);
CD = zeros(1,N_ss);
stage_ID = zeros(1,N_ss);



for ii = 1:length(state_transition_matrix)
    node_tag = node_names{ii}(end-1:end);
    for jj = 1:length(stage_indicators)
        if strcmp(node_tag,stage_indicators{jj}) == 1
            FN(ii) = FN_stages(jj);
            CD(ii) = CD_stages(jj);
            stage_ID(ii) = stage_numbers(jj);
        end
    end
end



%Define number of (training, testing) instances
num_training_set = 1;
num_testing_set = 1;
success_fail = zeros(1,num_training_set);
success_fail_std = zeros(1,num_training_set);
eta = 0.99;

for experiments = 1:num_training_set
    
    
    %%%%%%% Initialization of values (V_Data) and average reward values (rho_Data)
    V_Data{1} = zeros(2,N_ss); %{t}(DIFT:APT,state-1:state-N_ss)
    rho_Data{1} = zeros(2,1);  %{t}(DIFT:APT)
    
    
    %%% Initializing player policies and gradient data
    Policy_Data = cell(1); %{t}{state}{player}(action-1-player:action-end-player)
    Gradient_Data = cell(1); %{t}{state}{player}(action-1-player:action-end-player)
    
    for ii = 1:N_ss
        %Initializing player one policy
        num_act_DIFT = length(trap_set{ii})+1; %+1 for no trapping
        
        if num_act_DIFT > 0
            Policy_Data{1}{ii}{1} = ones(1,num_act_DIFT)./num_act_DIFT;
            Gradient_Data{1}{ii}{1} = zeros(1,num_act_DIFT);
        else
            Policy_Data{1}{ii}{1} = 1;
            Gradient_Data{1}{ii}{1} = 0;
        end
        
        %Initializing player two policy
        if isempty(find(APT_avoid == ii,1)) == 1
            if ii == s_0
                num_act_APT = length(find(state_transition_matrix(ii,:) == 1)); %No dropout only transition to entry points
            else
                num_act_APT = length(find(state_transition_matrix(ii,:) == 1)) + 1; %+1 for dropout
            end
            
            Policy_Data{1}{ii}{2} = ones(1,num_act_APT)./num_act_APT;
            Gradient_Data{1}{ii}{2} = zeros(1,num_act_APT);
        else
            Policy_Data{1}{ii}{2} = 1;
            Gradient_Data{1}{ii}{2} = 0;
        end
    end
    
    initial_state = s_0; %Initial state of the game
    current_state = initial_state;
    
    %Initializing variables that keep track of number of times each state, (state, DIFT_action), (state, APT_action) and (state, DIFT_action, APT_action) has been used
    State_Tracker = zeros(1,N_ss);
    State_Action_Tracker = cell(1);
    State_BothActions_Tracker = cell(1);
    
    for ii = 1:N_ss
        num_of_acts_DIFT = length(Policy_Data{1}{ii}{1});
        num_of_acts_APT = length(Policy_Data{1}{ii}{2});
        State_Action_Tracker{ii}{1} = zeros(1,num_of_acts_DIFT);
        State_Action_Tracker{ii}{2} = zeros(1,num_of_acts_APT);
        State_BothActions_Tracker{ii} = zeros(num_of_acts_DIFT,num_of_acts_APT);
    end
    
    t = 1;
    s_0_counter = 1; %For counting number of times s_0 visited
    counter = 1; %For counting number of iterations after 7000th iteration
    
    for episode = 1:episodes_sim
        
        reward_DIFT = 0; %DFIT's reward at the current episode
        reward_APT = 0; %APTs reward at the current episode
        %Finding the state ID
        current_state_ID = current_state;
        
        %Choosing actions for DIFT and APT based on their respective policies at the current iteration
        %DIFT
        DIFT_action_ID = act_choice(Policy_Data{t}{current_state_ID}{1});
        %APT
        APT_action_ID = act_choice(Policy_Data{t}{current_state_ID}{2});
        
        %Determining next state and reward of the game based on the next state of the game
        ss_entry_des_ID = zeros(1,length(ss_entry_dest));
        for nn = 1:length(ss_entry_des_ID)
            var1_cur = find(ss_entry_dest{nn} == current_state_ID);
            if isempty(var1_cur) == 1
                ss_entry_des_ID(nn) = -1;
            else
                ss_entry_des_ID(nn) = var1_cur;
            end
        end
        
        var2_cur = find(ss_entry_des_ID ~= -1);
        if isempty(var2_cur) == 1
            special_flag = -1;
        else
            special_flag = var2_cur;
        end
        
        if current_state_ID == s_0 %At pseudo node
            neighbor_IDs = find(state_transition_matrix(current_state_ID,:) == 1);
            next_state = neighbor_IDs(APT_action_ID);
        elseif special_flag ~= -1 %Entry_or_destination (special)
            if special_flag == 1 %Entry_Point
                neighbor_IDs = find(state_transition_matrix(current_state_ID,:) == 1);
                if length(neighbor_IDs) >= DIFT_action_ID %Decide to Trap at a neighbor
                    reward_DIFT = CD(current_state_ID);
                end
                
                if length(neighbor_IDs) >= APT_action_ID %NoDropout
                    if DIFT_action_ID == APT_action_ID
                        success = act_choice([FN(current_state_ID) 1-FN(current_state_ID)]);
                        if success == 2 %Detected
                            next_state = s_0;
                            reward_DIFT = reward_DIFT + DIFT_win(stage_ID(current_state_ID));
                            reward_APT = -APT_win(stage_ID(current_state_ID));
                        else %Miss ditection due to false negatives
                            next_state = neighbor_IDs(APT_action_ID);
                            dest_flag = find(ss_entry_dest{stage_ID(current_state_ID)+1} == next_state);
                            if isempty(dest_flag) == 1 %Not a destination
                            else %Destination
                                reward_DIFT = reward_DIFT + DIFT_lose(stage_ID(current_state_ID));
                                reward_APT = APT_win(stage_ID(current_state_ID));
                            end
                        end
                    else %Not detected (missed due to benign)
                        next_state = neighbor_IDs(APT_action_ID);
                        dest_flag = find(ss_entry_dest{stage_ID(current_state_ID)+1} == next_state);
                        if isempty(dest_flag) == 1 %Not a destination
                        else %Destination
                            reward_DIFT = reward_DIFT + DIFT_lose(stage_ID(current_state_ID));
                            reward_APT = APT_win(stage_ID(current_state_ID));
                        end
                    end
                else %Dropout
                    next_state = s_0;
                    reward_DIFT = reward_DIFT - 1*APT_drop(stage_ID(current_state_ID));
                    reward_APT = APT_drop(stage_ID(current_state_ID));
                end
                
            elseif special_flag == 2 %Stage 1 ----> Stage 2
                next_state = 8;
            elseif special_flag == 3 %Stage 2 ----> Stage 3
                next_state = 15;
            else %Final Stage ----> s_0
                next_state = s_0;
            end
            
        else %any normal state
            neighbor_IDs = find(state_transition_matrix(current_state_ID,:) == 1);
            if length(neighbor_IDs) >= DIFT_action_ID %Decide to Trap at a neighbor
                reward_DIFT = CD(current_state_ID);
            end
            
            if length(neighbor_IDs) >= APT_action_ID %NoDropout
                if DIFT_action_ID == APT_action_ID
                    success = act_choice([FN(current_state_ID) 1-FN(current_state_ID)]);
                    if success == 2 %Detected
                        next_state = s_0;
                        reward_DIFT = reward_DIFT + DIFT_win(stage_ID(current_state_ID));
                        reward_APT = -APT_win(stage_ID(current_state_ID));
                    else %Miss ditection due to false negatives
                        next_state = neighbor_IDs(APT_action_ID);
                        dest_flag = find(ss_entry_dest{stage_ID(current_state_ID)+1} == next_state);
                        if isempty(dest_flag) == 1 %Not a destination
                        else %Destination
                            reward_DIFT = reward_DIFT + DIFT_lose(stage_ID(current_state_ID));
                            reward_APT = APT_win(stage_ID(current_state_ID));
                        end
                    end
                else %Not detected (missed due to benign)
                    next_state = neighbor_IDs(APT_action_ID);
                    dest_flag = find(ss_entry_dest{stage_ID(current_state_ID)+1} == next_state);
                    if isempty(dest_flag) == 1 %Not a destination
                    else %Destination
                        reward_DIFT = reward_DIFT + DIFT_lose(stage_ID(current_state_ID));
                        reward_APT = APT_win(stage_ID(current_state_ID));
                    end
                end
            else %Dropout
                next_state = s_0;
                reward_DIFT = reward_DIFT - 1*APT_drop(stage_ID(current_state_ID));
                reward_APT = APT_drop(stage_ID(current_state_ID));
            end
        end
        
        next_state_ID = next_state;
        
        if DIFT_action_ID == 0
            DIFT_action_ID = 1;
        end
        
        if APT_action_ID == 0
            APT_action_ID = 1;
        end
        
        %%%%%%% Tracking the number of times each state, (state, DIFT_action), (state, APT_action) and (state, DIFT_action,APT_action) has been explored during the iterations
        State_Tracker(current_state_ID) = State_Tracker(current_state_ID) + 1;
        State_Action_Tracker{current_state_ID}{1}(DIFT_action_ID) = State_Action_Tracker{current_state_ID}{1}(DIFT_action_ID) + 1;
        State_Action_Tracker{current_state_ID}{2}(APT_action_ID) = State_Action_Tracker{current_state_ID}{2}(APT_action_ID) + 1;
        State_BothActions_Tracker{current_state_ID}(DIFT_action_ID,APT_action_ID) = State_BothActions_Tracker{current_state_ID}(DIFT_action_ID,APT_action_ID) + 1;
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %                        MA-ARNE Algorithm                        %
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        
        %%%%%%%%%%%%%%%%%%%% Defining learning rates %%%%%%%%%%%%%%%%%%%%%%
        if episode < 7000
            gamma_1 = 1;  %learning rate for average reward
            gamma_2 = 0.5; %learning rate for V updates and Gradient updates
            gamma_3 = 1; %learning rate for policy updates
        else
            
            counter = counter + 1;
            
            gamma_1 = 1/(1 + counter*log(counter));  %learning rate for average reward
            gamma_2 = 1.6/(State_Tracker(current_state_ID)); %learning rate for V updates and Gradient updates
            gamma_3 = 1/counter; %learning rate for policy updates
            
        end
        
        %%%%%%%%%%%%%%%%%%%%    Update Equations    %%%%%%%%%%%%%%%%%%%%%%
        
        
        %%%% Average Reward Updates
        rho_Data{t+1}(1) = (1-gamma_1)*rho_Data{t}(1) + gamma_1*(((t-1)*rho_Data{t}(1) + reward_DIFT)/(t));
        rho_Data{t+1}(2) = (1-gamma_1)*rho_Data{t}(2) + gamma_1*(((t-1)*rho_Data{t}(2) + reward_APT)/(t));
        
        %Update value functions (V)
        V_Data{t+1} = V_Data{t};
        V_Data{t+1}(1,current_state_ID) = (1-gamma_2)*V_Data{t}(1,current_state_ID) + gamma_2*(reward_DIFT - rho_Data{t+1}(1) + eta*V_Data{t}(1,next_state_ID));
        V_Data{t+1}(2,current_state_ID) = (1-gamma_2)*V_Data{t}(2,current_state_ID) + gamma_2*(reward_APT - rho_Data{t+1}(2) + eta*V_Data{t}(2,next_state_ID));
        
        
        %Gradient updates
        
        V_term_DIFT = reward_DIFT - rho_Data{t+1}(1) + V_Data{t}(1,next_state_ID) - V_Data{t}(1,current_state_ID);
        V_term_APT = reward_APT - rho_Data{t+1}(2) + V_Data{t}(2,next_state_ID) - V_Data{t}(2,current_state_ID);
        
        Gradient_Data{t+1} = Gradient_Data{t};
        Gradient_Data{t+1}{current_state_ID}{1}(DIFT_action_ID) = (1-gamma_2)*Gradient_Data{t}{current_state_ID}{1}(DIFT_action_ID) + gamma_2*(V_term_DIFT+V_term_APT);
        Gradient_Data{t+1}{current_state_ID}{2}(APT_action_ID) = (1-gamma_2)*Gradient_Data{t}{current_state_ID}{2}(APT_action_ID) + gamma_2*(V_term_DIFT+V_term_APT);
        
        %policy updates
        Policy_Data{t+1} = Policy_Data{t};
        Policy_Data{t+1}{current_state_ID}{1}(DIFT_action_ID) = Policy_Data{t}{current_state_ID}{1}(DIFT_action_ID) - gamma_3*sqrt(Policy_Data{t}{current_state_ID}{1}(DIFT_action_ID))*abs(V_term_DIFT)*sign(-1*Gradient_Data{t+1}{current_state_ID}{1}(DIFT_action_ID));
        Policy_Data{t+1}{current_state_ID}{2}(APT_action_ID) = Policy_Data{t}{current_state_ID}{2}(APT_action_ID) - gamma_3*sqrt(Policy_Data{t}{current_state_ID}{2}(APT_action_ID))*abs(V_term_APT)*sign(-1*Gradient_Data{t+1}{current_state_ID}{2}(APT_action_ID));
        
        %Projecting policies to probability simplex
        Policy_Data{t+1}{current_state_ID}{1} = policy_projection(Policy_Data{t+1}{current_state_ID}{1});
        Policy_Data{t+1}{current_state_ID}{2} = policy_projection(Policy_Data{t+1}{current_state_ID}{2});
        
        
        %To improve the exploration in the algorithm: Add a decaying noise to policies of both DIFT and APT when each of them converge to a deterministic policy
        deterministic_entry_DIFT = find(Policy_Data{t+1}{current_state_ID}{1} >= 0.95);
        deterministic_entry_APT = find(Policy_Data{t+1}{current_state_ID}{2} >= 0.95);
        
        if episode > 10000
            noise = 0.01;
        elseif episode > 5000
            noise = 0.1;
        elseif episode > 2500
            noise = 0.2;
        else
            noise = 0.3;
        end
        
        if isempty(deterministic_entry_DIFT) == 0
            for ee = 1:length(Policy_Data{t+1}{current_state_ID}{1})
                if ee ~= deterministic_entry_DIFT
                    Policy_Data{t+1}{current_state_ID}{1}(ee) =  Policy_Data{t+1}{current_state_ID}{1}(ee) + noise;
                end
            end
            Policy_Data{t+1}{current_state_ID}{1} = policy_projection(Policy_Data{t+1}{current_state_ID}{1});
        end
        
        if isempty(deterministic_entry_APT) == 0
            for ee = 1:length(Policy_Data{t+1}{current_state_ID}{2})
                if ee ~= deterministic_entry_APT
                    Policy_Data{t+1}{current_state_ID}{2}(ee) =  Policy_Data{t+1}{current_state_ID}{2}(ee) + noise;
                end
            end
            Policy_Data{t+1}{current_state_ID}{2} = policy_projection(Policy_Data{t+1}{current_state_ID}{2});
        end
        
        
        if next_state == s_0 %Counting number of times s_0 is reached as a next state (one round of game is done: i.e., APT dropped out, APT reach the final goal(destination), or DIFT detects APT)
            s_0_counter = s_0_counter + 1;
        end
        if (mod(s_0_counter,100) == 0) %Ranomly intialize the game to improve the exploration (every 100th time s_0 is visited)
            next_state = random_initial_state(State_Tracker,ss_entry_dest,1:N_ss);
            s_0_counter = s_0_counter + 1;
        end
        
        current_state = next_state;
        t = t + 1;
        sprintf('Experiment num: %d, Episode: %d', experiments,episode)
    end
    clc
end
end
