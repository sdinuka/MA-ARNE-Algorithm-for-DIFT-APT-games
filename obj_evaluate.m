 function [obj_value,obj_def, obj_adv] = obj_evaluate(rho_data,V_Data,Policy_Data,tt,N_ss,stage_ID,ss_entry_dest,CD,APT_win,APT_drop,state_transition_matrix,FN)
obj_value = 0;

obj_def = 0;
obj_adv = 0;

%Evaluating obj. function corresponding to defender
for ii = 1:N_ss
    r_term = 0;
    PV_term = 0;
    for dd = 1:length(Policy_Data{tt}{ii}{1})
        player_index = 1;
        player_action_index = dd;
        current_state = ii;
        [r_tmp, PV_tmp] = r_PV_eval(player_index,player_action_index,current_state,V_Data,Policy_Data,tt,stage_ID,ss_entry_dest,CD,APT_win,APT_drop,state_transition_matrix,FN,N_ss);
        r_term = r_term + r_tmp*Policy_Data{tt}{ii}{1}(dd);
        PV_term = PV_term + PV_tmp*Policy_Data{tt}{ii}{1}(dd);
    end
    obj_def = obj_def + (rho_data(1) + V_Data{tt}(1,ii) - r_term - PV_term);
end


%Evaluating obj. function corresponding to adversary
for ii = 1:N_ss
    r_term = 0;
    PV_term = 0;
    for aa = 1:length(Policy_Data{tt}{ii}{2})
        player_index = 2;
        player_action_index = aa;
        current_state = ii;
        [r_tmp, PV_tmp] = r_PV_eval(player_index,player_action_index,current_state,V_Data,Policy_Data,tt,stage_ID,ss_entry_dest,CD,APT_win,APT_drop,state_transition_matrix,FN,N_ss);
        r_term = r_term + r_tmp*Policy_Data{tt}{ii}{2}(aa);
        PV_term = PV_term + PV_tmp*Policy_Data{tt}{ii}{2}(aa);
    end
    obj_adv = obj_adv + (rho_data(2) + V_Data{tt}(2,ii) - r_term - PV_term);
end

obj_value = obj_adv + obj_def;

end