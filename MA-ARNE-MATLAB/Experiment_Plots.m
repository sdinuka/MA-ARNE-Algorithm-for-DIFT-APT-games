%Ploting the convergence of the algorithm
function Experiment_Plots(t, Policy_Data, V_Data, N_ss, stage_ID, ss_entry_dest, FN, CD, APT_drop, APT_win, DIFT_win,DIFT_lose,state_transition_matrix, trap_set)
clc
import java.util.ArrayList

counter = 1;
for tt = 1:500:t
    [Avg_reward_p1,Avg_reward_p2,action_set_DIFT,action_set_APT,state_set] = find_average_reward(Policy_Data,tt,N_ss,stage_ID,ss_entry_dest,CD,APT_win,APT_drop,state_transition_matrix,FN);
    rho_data = [Avg_reward_p1(end); Avg_reward_p2(end)];
    rho_data_plot(:,counter) = rho_data;
    [obj_value,obj_def, obj_adv] = obj_evaluate(rho_data,V_Data,Policy_Data,tt,N_ss,stage_ID,ss_entry_dest,CD,APT_win,APT_drop,state_transition_matrix,FN);
    obj_value_plot(counter) = obj_value;
    obj_value_DIFT_plot(counter) = obj_def;
    obj_value_APT_plot(counter) = obj_adv;
    value_DIFT_plot(counter) = mean(V_Data{tt}(1,:));
    value_APT_plot(counter) = mean(V_Data{tt}(2,:));
    counter = counter + 1;
end

%Convergence of the algorithm
figure(1)
plot(obj_value_plot(1:end-1),'b','LineWidth',2);

hold on
plot(obj_value_DIFT_plot(1:end-1),'g','LineWidth',2)
p3 = plot(obj_value_APT_plot(1:end-1),'r','LineWidth',2);
p3.MarkerSize = 0.5;
yline(0,'--k')
hold off
ax = gca;
set(ax,'FontSize',20,'FontWeight','bold')
xlabel('Iteration \times 10^{6}')
ylim([-250 350])
xlim([0 250])
set(gca,'XTickLabel',0:0.5:2.5)
ylabel('Objective values')
legend('\phi_{T}', '\phi_{D}', '\phi_{A}')


%Convergence of average reward valeus of the players
figure(2)
plot(rho_data_plot(1,:),'g','LineWidth',2)
hold on
plot(rho_data_plot(2,:),'r','LineWidth',2)
hold off
ax = gca;
set(ax,'FontSize',20,'FontWeight','bold')
xlabel('Iteration \times 10^{6}')
ylim([-15 15])
xlim([0 250])
set(gca,'XTickLabel',0:0.5:2.5)
ylabel('Average reward values')
legend('\rho_{D}', '\rho_{A}')

bar_Data = zeros(3,2);
[Avg_reward_p1,Avg_reward_p2,action_set_DIFT,action_set_APT,state_set] = find_average_reward(Policy_Data,t-1,N_ss,stage_ID,ss_entry_dest,CD,APT_win,APT_drop,state_transition_matrix,FN);
bar_Data(1,1) = Avg_reward_p1(end);
bar_Data(1,2) = Avg_reward_p2(end);

%%% Defining Uniform DIFT policy and ANRE APT policy
Policy_Data_uniform = cell(1); %{t}{state}{player}(action-1-player:action-end-player)
for ii = 1:N_ss
    %Defining DIFT policy
    num_act_p1 = length(trap_set{ii})+1; %+1 for no trapping
    
    if num_act_p1 > 1
        Policy_Data_uniform{1}{ii}{1} = zeros(1,num_act_p1);
        Policy_Data_uniform{1}{ii}{1}(1:num_act_p1) = ones(1,num_act_p1)./(num_act_p1);
    else
        Policy_Data_uniform{1}{ii}{1} = 1;
    end
    
    %ANRE APT policy
    Policy_Data_uniform{1}{ii}{2} = Policy_Data{t-1}{ii}{2};
    
end

[Avg_reward_p1_uni,Avg_reward_p2_uni,action_set_DIFT,action_set_APT,state_set] = find_average_reward(Policy_Data_uniform,1,N_ss,stage_ID,ss_entry_dest,CD,APT_win,APT_drop,state_transition_matrix,FN);
bar_Data(2,1) = Avg_reward_p1_uni(end);
bar_Data(2,2) = Avg_reward_p2_uni(end);

%%% Defining Uniform DIFT policy and ANRE APT policy
Policy_Data_target = cell(1); %{t}{state}{player}(action-1-player:action-end-player)
targets = [3 10 17];
for ii = 1:N_ss
    %Defining DIFT policy
    num_act_p1 = length(trap_set{ii})+1; %+1 for no trapping
    
    if num_act_p1 > 1
        Policy_Data_target{1}{ii}{1} = zeros(1,num_act_p1);
        
        neighbor_IDs = find(state_transition_matrix(ii,:) == 1);
        target_ID = [];
        for jj = 1:length(targets)
            target_ID_tmp = find(neighbor_IDs == targets(jj));
            if isempty(target_ID_tmp) == 0
                target_ID = target_ID_tmp;
            end
        end
        if isempty(target_ID) == 1 %Non taregt --> No trap
            Policy_Data_target{1}{ii}{1}(end) = 1;
        else
            Policy_Data_target{1}{ii}{1}(target_ID) = 1;
        end
    else
        Policy_Data_target{1}{ii}{1} = 1;
    end
    
    %ANRE APT policy
    Policy_Data_target{1}{ii}{2} = Policy_Data{t-1}{ii}{2};
end

[Avg_reward_p1_tar,Avg_reward_p2_tar,action_set_DIFT,action_set_APT,state_set] = find_average_reward(Policy_Data_target,1,N_ss,stage_ID,ss_entry_dest,CD,APT_win,APT_drop,state_transition_matrix,FN);
bar_Data(3,1) = Avg_reward_p1_tar(end);
bar_Data(3,2) = Avg_reward_p2_tar(end);


figure(3)
set(gcf, 'Position',  [100, 100, 1000, 500])
bar(bar_Data)
hold on
yline(bar_Data(1,1),'--k')
hold off
ax = gca;
set(ax,'FontSize',20,'FontWeight','bold')
%xlabel('Word ID','FontSize',12,'FontWeight','bold')
ylabel('Average reward values')
legend('DIFT policy', 'APT policy')



%State Space plot
figure(4)
SS = digraph(state_transition_matrix);
SS_plot = plot(SS);
title('State space of DIFT-APT game on Ransomware attack data')

end