function [Avg_reward_p1,Avg_reward_p2,action_set_DIFT,action_set_APT,state_set] = find_average_reward(Policy_Data,t,N_ss,stage_ID,ss_entry_dest,CD,APT_win,APT_drop,state_transition_matrix,FN)

T_sim = 10000;
Avg_reward_p1 = zeros(1,T_sim);
Avg_reward_p2 = zeros(1,T_sim);
action_set_DIFT = zeros(1,T_sim);
action_set_APT = zeros(1,T_sim);
state_set = zeros(1,T_sim);
s_0 = N_ss;
current_state = s_0;
for ii = 1:T_sim
    
    reward_p1 = 0; %Player one's reward (DIFT)
    reward_p2 = 0; %Player two's reward (APT)
    %Finding the state ID
    current_state_ID = current_state;
    
    %Choosing actions correpsonding next grid positions
    %Player 1
    P1_action_ID = act_choice(Policy_Data{t}{current_state_ID}{1});
    %Player 2
    P2_action_ID = act_choice(Policy_Data{t}{current_state_ID}{2});
    neighbor_set_ID = find(state_transition_matrix(current_state_ID,:) == 1);
    if P1_action_ID > length(neighbor_set_ID) %No trap 
        action_set_DIFT(ii) = 0;
    else
        action_set_DIFT(ii) = neighbor_set_ID(P1_action_ID);
    end
    
    if P2_action_ID > length(neighbor_set_ID) %No trap 
        action_set_APT(ii) = 0;
    else
        action_set_APT(ii) = neighbor_set_ID(P2_action_ID);
    end
    
    if current_state_ID == s_0
        state_set(ii) = 0;
    else
        state_set(ii) = current_state_ID;
    end
    
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
        next_state = neighbor_IDs(P2_action_ID);
    elseif special_flag ~= -1 %Entry_or_destination (special)
        if special_flag == 1 %Entry_Point
            neighbor_IDs = find(state_transition_matrix(current_state_ID,:) == 1);
            if length(neighbor_IDs) >= P1_action_ID %Decide to Trap at a neighbor
                reward_p1 = CD(neighbor_IDs(P1_action_ID));
            end
            
            if length(neighbor_IDs) >= P2_action_ID %NoDropout
                if P1_action_ID == P2_action_ID
                    success = act_choice([FN(current_state_ID) 1-FN(current_state_ID)]);
                    if success == 2 %Detected
                        next_state = s_0;
                        reward_p1 = reward_p1 + APT_win(stage_ID(current_state_ID));
                        reward_p2 = -APT_win(stage_ID(current_state_ID));
                    else %Miss ditection due to false negatives
                        next_state = neighbor_IDs(P2_action_ID);
                        dest_flag = find(ss_entry_dest{stage_ID(current_state_ID)+1} == next_state);
                        if isempty(dest_flag) == 1 %Not a destination
                        else %Destination
                            reward_p1 = reward_p1 - APT_win(stage_ID(current_state_ID));
                            reward_p2 = APT_win(stage_ID(current_state_ID));
                        end
                    end
                else %Not detected (missed due to benign)
                    next_state = neighbor_IDs(P2_action_ID);
                    dest_flag = find(ss_entry_dest{stage_ID(current_state_ID)+1} == next_state);
                    if isempty(dest_flag) == 1 %Not a destination
                    else %Destination
                        reward_p1 = reward_p1 - APT_win(stage_ID(current_state_ID));
                        reward_p2 = APT_win(stage_ID(current_state_ID));
                    end
                end
            else %Dropout
                next_state = s_0;
                reward_p1 = reward_p1 - 1*APT_drop(stage_ID(current_state_ID));
                reward_p2 = APT_drop(stage_ID(current_state_ID));
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
        if length(neighbor_IDs) >= P1_action_ID %Decide to Trap at a neighbor
            reward_p1 = CD(neighbor_IDs(P1_action_ID));
        end
        
        if length(neighbor_IDs) >= P2_action_ID %NoDropout
            if P1_action_ID == P2_action_ID
                success = act_choice([FN(current_state_ID) 1-FN(current_state_ID)]);
                if success == 2 %Detected
                    next_state = s_0;
                    reward_p1 = reward_p1 + APT_win(stage_ID(current_state_ID));
                    reward_p2 = -APT_win(stage_ID(current_state_ID));
                else %Miss ditection due to false negatives
                    next_state = neighbor_IDs(P2_action_ID);
                    dest_flag = find(ss_entry_dest{stage_ID(current_state_ID)+1} == next_state);
                    if isempty(dest_flag) == 1 %Not a destination
                    else %Destination
                        reward_p1 = reward_p1 - APT_win(stage_ID(current_state_ID));
                        reward_p2 = APT_win(stage_ID(current_state_ID));
                    end
                end
            else %Not detected (missed due to benign)
                next_state = neighbor_IDs(P2_action_ID);
                dest_flag = find(ss_entry_dest{stage_ID(current_state_ID)+1} == next_state);
                if isempty(dest_flag) == 1 %Not a destination
                else %Destination
                    reward_p1 = reward_p1 - APT_win(stage_ID(current_state_ID));
                    reward_p2 = APT_win(stage_ID(current_state_ID));
                end
            end
        else %Dropout
            next_state = s_0;
            reward_p1 = reward_p1 - 1*APT_drop(stage_ID(current_state_ID));
            reward_p2 = APT_drop(stage_ID(current_state_ID));
        end
    end
    
    current_state = next_state;
    
    Avg_reward_p1(ii+1) = ((ii-1)*Avg_reward_p1(ii) + reward_p1)/(ii);
    Avg_reward_p2(ii+1) = ((ii-1)*Avg_reward_p2(ii) + reward_p2)/(ii);
    
    
    
    
end


end