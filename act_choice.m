%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%  Choosing an action given the action set's probability distribution %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Method:
%   (1) Sort the probability distribution in ascending order
%   (2) Build the CDF of the probability distribution function
%   (3) Choose a unifrom random number between 0 and 1 according to a uniform distribution
%   (4) Depending on the place where random number generated fall in the CDF select the corresponding action

function chosen_act_k = act_choice(P_tk_t) %***
chosen_act_k = 0;
num_act_k = length(P_tk_t);
if num_act_k > 0
    [P_temp, I] = sort(P_tk_t);
    P(1) = P_temp(1);
    for j = 1:(num_act_k-1)
        P(j+1) = sum(P_temp(1:j+1));
    end
    %P
    choice = rand(1,1);
    for l = 1:num_act_k
        if l == 1
            lb = 0;
        else
            lb = P(l-1);
        end
        if (lb <= choice) && (choice <= P(l))
            chosen_act_k = I(l);
        end
    end
end 
end