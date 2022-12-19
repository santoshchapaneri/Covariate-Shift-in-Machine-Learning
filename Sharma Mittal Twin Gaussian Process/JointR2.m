function [R2, MSE] = JointR2(TGPTarget, TestTarget)
% Compute the joint R2 value of V and A

D = size(TestTarget,2);
R2 = zeros(1,D); MSE = zeros(1,D);
for i = 1:D
    [R2(i), MSE(i)] = rsquare(TestTarget(:,i),TGPTarget(:,i),true);
end
% R2 = mean(R2);
% MSE = mean(MSE);