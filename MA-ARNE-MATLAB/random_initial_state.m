function [initial_state] = random_initial_state(State_Tracker,entry_dest_info,state_space)

avoid_states = [];
for ii = 2:length(entry_dest_info)%Collect all destination related states
    avoid_states = [avoid_states entry_dest_info{ii}];
end
avoid_states = [avoid_states length(state_space)]; %Avoid choosing s_0 as next state

flag = 0;
[~, Index] = sort(State_Tracker);
counter = 1;
while flag == 0
    state = state_space(Index(counter));
    avoid_flag = find(avoid_states == state);
    if isempty(avoid_flag) == 1 %Chosen state is not a avoid state
        flag = 1;
    end
    counter = counter + 1;
end
initial_state = state;
end