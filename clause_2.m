clc; clear;
classifiers = zeros(28*28+1, 10);
for digit=0:9    
    x = clause_2_general(digit);
    classifiers(:,digit+1) = x;   
end
%% ==================== New Test Set for Clause 3 ====================
% N_test = test.count;
% A_new_test = reshape(test.images,N_test,28*28);
% A_new_test = [A_new_test, ones(N_test,1)];
% true_labels = test.labels;