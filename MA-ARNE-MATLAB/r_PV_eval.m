function [r_term, PV_term] = r_PV_eval(player_index,player_action_index,current_state,V_Data,Policy_Data,t,stage_ID,ss_entry_dest,CD,APT_win,APT_drop,state_transition_matrix,FN,N_ss)
s_0 = N_ss;
r_term = 0;
PV_term = 0;
current_state_ID = current_state;
if player_index == 1 %Fix player one action average over player two policy
    for aa = 1:length(Policy_Data{t}{current_state}{2})
        reward_p1 = 0;
        PV_p1 = 0;
        
        %Choosing actions correpsonding next grid positions
        %Player 1
        P1_action_ID = player_action_index;
        %Player 2
        P2_action_ID = aa;
        
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
            PV_p1 = V_Data{t}(1,next_state);
        elseif special_flag ~= -1 %Entry_or_destination (special)
            if special_flag == 1 %Entry_Point
                neighbor_IDs = find(state_transition_matrix(current_state_ID,:) == 1);
                if length(neighbor_IDs) >= P1_action_ID %Decide to Trap at a neighbor
                    reward_p1 = CD(neighbor_IDs(P1_action_ID));
                end
                
                if length(neighbor_IDs) >= P2_action_ID %NoDropout
                    if P1_action_ID == P2_action_ID %Decides to trap
                        
                        for success = 1:2
                            if success == 2 %Detected
                                next_state = s_0;
                                reward_p1 = reward_p1 + APT_win(stage_ID(current_state_ID))*(1-FN(current_state_ID));
                                PV_p1 = PV_p1 + (V_Data{t}(1,next_state))*(1-FN(current_state_ID));
                            else %Miss ditection due to false negatives
                                next_state = neighbor_IDs(P2_action_ID);
                                dest_flag = find(ss_entry_dest{stage_ID(current_state_ID)+1} == next_state);
                                if isempty(dest_flag) == 1 %Not a destination
                                else %Destination
                                    reward_p1 = reward_p1 - APT_win(stage_ID(current_state_ID))*FN(current_state_ID);
                                end
                                PV_p1 = PV_p1 + (V_Data{t}(1,next_state))*FN(current_state_ID);
                            end
                        end
                        
                    else %Not detected (missed due to benign)
                        next_state = neighbor_IDs(P2_action_ID);
                        dest_flag = find(ss_entry_dest{stage_ID(current_state_ID)+1} == next_state);
                        if isempty(dest_flag) == 1 %Not a destination
                        else %Destination
                            reward_p1 = reward_p1 - APT_win(stage_ID(current_state_ID));
                        end
                        PV_p1 = V_Data{t}(1,next_state);
                    end
                    
                else %Dropout
                    next_state = s_0;
                    reward_p1 = reward_p1 - 1*APT_drop(stage_ID(current_state_ID));
                    PV_p1 = V_Data{t}(1,next_state);
                end
                
            elseif special_flag == 2 %Stage 1 ----> Stage 2
                next_state = 8;
                PV_p1 = V_Data{t}(1,next_state);
            elseif special_flag == 3 %Stage 2 ----> Stage 3
                next_state = 15;
                PV_p1 = V_Data{t}(1,next_state);
            else %Final Stage ----> s_0
                next_state = s_0;
                PV_p1 = V_Data{t}(1,next_state);
            end
        else %any normal state
            neighbor_IDs = find(state_transition_matrix(current_state_ID,:) == 1);
            if length(neighbor_IDs) >= P1_action_ID %Decide to Trap at a neighbor
                reward_p1 = CD(neighbor_IDs(P1_action_ID));
            end
            
            if length(neighbor_IDs) >= P2_action_ID %NoDropout
                if P1_action_ID == P2_action_ID %Decides to trap
                    
                    for success = 1:2
                        if success == 2 %Detected
                            next_state = s_0;
                            reward_p1 = reward_p1 + APT_win(stage_ID(current_state_ID))*(1-FN(current_state_ID));
                            PV_p1 = PV_p1 + (V_Data{t}(1,next_state))*(1-FN(current_state_ID));
                        else %Miss ditection due to false negatives
                            next_state = neighbor_IDs(P2_action_ID);
                            dest_flag = find(ss_entry_dest{stage_ID(current_state_ID)+1} == next_state);
                            if isempty(dest_flag) == 1 %Not a destination
                            else %Destination
                                reward_p1 = reward_p1 - APT_win(stage_ID(current_state_ID))*FN(current_state_ID);
                            end
                            PV_p1 = PV_p1 + (V_Data{t}(1,next_state))*FN(current_state_ID);
                        end
                    end
                    
                else %Not detected (missed due to benign)
                    next_state = neighbor_IDs(P2_action_ID);
                    dest_flag = find(ss_entry_dest{stage_ID(current_state_ID)+1} == next_state);
                    if isempty(dest_flag) == 1 %Not a destination
                    else %Destination
                        reward_p1 = reward_p1 - APT_win(stage_ID(current_state_ID));
                    end
                    PV_p1 = V_Data{t}(1,next_state);
                end
                
            else %Dropout
                next_state = s_0;
                reward_p1 = reward_p1 - 1*APT_drop(stage_ID(current_state_ID));
                PV_p1 = V_Data{t}(1,next_state);
            end
        end
        
        
        r_term = r_term + reward_p1*Policy_Data{t}{current_state}{2}(aa);
        PV_term = PV_term + PV_p1*Policy_Data{t}{current_state}{2}(aa);

    end
    
elseif player_index == 2 %Fix player two action average over player one policy
    for dd = 1:length(Policy_Data{t}{current_state}{1})
        reward_p2 = 0;
        PV_p2 = 0;
        
        %Choosing actions correpsonding next grid positions
        %Player 1
        P1_action_ID = dd;
        %Player 2
        P2_action_ID = player_action_index;
        
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
            PV_p2 = V_Data{t}(2,next_state);
        elseif special_flag ~= -1 %Entry_or_destination (special)
            if special_flag == 1 %Entry_Point
                neighbor_IDs = find(state_transition_matrix(current_state_ID,:) == 1);
                
                if length(neighbor_IDs) >= P2_action_ID %NoDropout
                    if P1_action_ID == P2_action_ID %Decides to trap
                        
                        for success = 1:2
                            if success == 2 %Detected
                                next_state = s_0;
                                reward_p2 = reward_p2 - APT_win(stage_ID(current_state_ID))*(1-FN(current_state_ID));
                                PV_p2 = PV_p2 + (V_Data{t}(2,next_state))*(1-FN(current_state_ID));
                            else %Miss ditection due to false negatives
                                next_state = neighbor_IDs(P2_action_ID);
                                dest_flag = find(ss_entry_dest{stage_ID(current_state_ID)+1} == next_state);
                                if isempty(dest_flag) == 1 %Not a destination
                                else %Destination
                                    reward_p2 = reward_p2 + APT_win(stage_ID(current_state_ID))*FN(current_state_ID);
                                end
                                PV_p2 = PV_p2 + (V_Data{t}(2,next_state))*FN(current_state_ID);
                            end
                        end
                        
                    else %Not detected (missed due to benign)
                        next_state = neighbor_IDs(P2_action_ID);
                        dest_flag = find(ss_entry_dest{stage_ID(current_state_ID)+1} == next_state);
                        if isempty(dest_flag) == 1 %Not a destination
                        else %Destination
                            reward_p2 = APT_win(stage_ID(current_state_ID));
                        end
                        PV_p2 = V_Data{t}(2,next_state);
                    end
                    
                else %Dropout
                    next_state = s_0;
                    reward_p2 = APT_drop(stage_ID(current_state_ID));
                    PV_p2 = V_Data{t}(2,next_state);
                end
                
            elseif special_flag == 2 %Stage 1 ----> Stage 2
                next_state = 8;
                PV_p2 = V_Data{t}(2,next_state);
            elseif special_flag == 3 %Stage 2 ----> Stage 3
                next_state = 15;
                PV_p2 = V_Data{t}(2,next_state);
            else %Final Stage ----> s_0
                next_state = s_0;
                PV_p2 = V_Data{t}(2,next_state);
            end
        else %any normal state
            neighbor_IDs = find(state_transition_matrix(current_state_ID,:) == 1);
            
            if length(neighbor_IDs) >= P2_action_ID %NoDropout
                if P1_action_ID == P2_action_ID %Decides to trap
                    
                    for success = 1:2
                        if success == 2 %Detected
                            next_state = s_0;
                            reward_p2 = reward_p2 -APT_win(stage_ID(current_state_ID))*(1-FN(current_state_ID));
                            PV_p2 = PV_p2 + (V_Data{t}(2,next_state))*(1-FN(current_state_ID));
                        else %Miss ditection due to false negatives
                            next_state = neighbor_IDs(P2_action_ID);
                            dest_flag = find(ss_entry_dest{stage_ID(current_state_ID)+1} == next_state);
                            if isempty(dest_flag) == 1 %Not a destination
                            else %Destination
                                reward_p2 = reward_p2 + APT_win(stage_ID(current_state_ID))*FN(current_state_ID);
                            end
                            PV_p2 = PV_p2 + (V_Data{t}(2,next_state))*FN(current_state_ID);
                        end
                    end
                    
                else %Not detected (missed due to benign)
                    next_state = neighbor_IDs(P2_action_ID);
                    dest_flag = find(ss_entry_dest{stage_ID(current_state_ID)+1} == next_state);
                    if isempty(dest_flag) == 1 %Not a destination
                    else %Destination
                        reward_p2 = APT_win(stage_ID(current_state_ID));
                    end
                    PV_p2 = V_Data{t}(2,next_state);
                end
                
            else %Dropout
                next_state = s_0;
                reward_p2 = APT_drop(stage_ID(current_state_ID));
                PV_p2 = V_Data{t}(2,next_state);
            end
        end
        
        r_term = r_term + reward_p2*Policy_Data{t}{current_state}{1}(dd);
        PV_term = PV_term + PV_p2*Policy_Data{t}{current_state}{1}(dd);
        
    end
else
    sprintf('Wrong Player index')
end

end