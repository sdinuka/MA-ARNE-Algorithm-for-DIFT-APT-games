function [policy_projected] = policy_projection(policy)

% if t < 0
% policy = policy + ((1/t.^2))*randn(1,length(policy));
% else
%     policy = policy + 0.1*randn(1,length(policy));
% end
% 
% exp_entries = zeros(1,length(policy));
% 
% for ii = 1:length(policy)
%     exp_entries(ii) = exp(policy(ii));
% end
% 
% policy_projected = exp_entries/sum(exp_entries);

%policy_exp = exp_entries/sum(exp_entries);


u = sort(policy,'descend');

max_index = 0;
find_max_index_flag = 0;

while find_max_index_flag == 0
    temp_max_val = u(max_index+1)+(1/(max_index+1))*(1-sum(u(1:(max_index+1))));
    if temp_max_val > 0
        max_index = max_index + 1;
    else
        find_max_index_flag = 1;
    end
    
    if max_index == length(policy)
        find_max_index_flag = 1;
    end
end

lambda = (1/max_index)*(1-sum(u(1:max_index)));
p = zeros(1,length(policy));
for ii = 1:length(policy)
    p(ii) = max([policy(ii)+lambda,0]);
end
policy_projected = p;
end