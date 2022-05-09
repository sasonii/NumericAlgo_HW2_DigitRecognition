clc; clear;
classifiers = zeros(28*28+1, 10);
for digit=0:9    
    x = clause_2_general(digit);
    classifiers(:,digit+1) = x;   
end
save('classifiers.mat','classifiers');