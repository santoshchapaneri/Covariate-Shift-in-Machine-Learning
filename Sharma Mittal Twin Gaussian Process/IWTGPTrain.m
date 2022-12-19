function [InvIKW, InvOKW, IKW, OKW] = IWTGPTrain(Input, Target, Param, W)
% Train Twin Gaussian Processes

IK = EvalKernel(Input, Input, 'rbf', Param.kparam1);
IKW = W*IK*W;
InvIKW = inv(IKW + Param.lambda*eye(size(IKW)));
IKW = IKW + Param.lambda*eye(size(IKW));

OK = EvalKernel(Target, Target, 'rbf', Param.kparam2);
OKW = W*OK*W;
InvOKW = inv(OKW + Param.lambda*eye(size(OKW)));
OKW = OKW + Param.lambda*eye(size(OKW));