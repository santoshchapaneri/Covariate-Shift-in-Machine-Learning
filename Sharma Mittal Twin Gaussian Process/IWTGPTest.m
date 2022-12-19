function TGPTarget = IWTGPTest(TestInput, Input, Target, Param, InvIKW, InvOKW, W)
% Make the prediction using Twin Gaussian Processess%
%CI = -Param.SMAlpha*(1 -Param.SMBeta)/ (2*(1-Param.SMAlpha) );
%CO = - (1 -Param.SMBeta)/2;
%CIO = - (1 -Param.SMBeta)/(2*(1-Param.SMAlpha));

T = size(TestInput,1);
TGPTarget = zeros(T,size(Target,2));
Weight = LinearRegressor(Input, Target);
for frame = 1:T
%     fprintf('%d\n',frame);
    % Initialize
    OneTestInput = TestInput(frame,:);
    EOneTestInput = [1 OneTestInput];
    InitTarget = (EOneTestInput*Weight)';

    % optimize
    IR = EvalKernel(Input,OneTestInput,'rbf',Param.kparam1);
    IRW = W*IR;
    % alpha is u_x, beta is eta_x
    alphaW = InvIKW*IRW; 
    betaW = (1+Param.lambda) - IRW'*InvIKW*IRW;
    
    Y = ComputeOutput(InitTarget, Target, Param.kparam2, Param.lambda, alphaW, betaW, InvOKW, W);
    TGPTarget(frame,:) = Y';
end

function [Y, fval] = ComputeOutput(Y, Target, kernelparam, lambda, alphaW, betaW, InvOKW, W)

%% Compute the output

options = optimset('GradObj','on');
options = optimset(options,'LargeScale','off');
options = optimset(options,'DerivativeCheck','off');
options = optimset(options,'Display','off');
% options = optimset(options,'Display','final');
options = optimset(options,'MaxIter',50);
options = optimset(options,'TolFun',1e-6);
options = optimset(options,'TolX',1e-6);
%options = optimset(options,'LineSearchType','cubicpoly');
aaa = sum(Target.^2,2);

% Optimization
[Y, fval] = minFunc(@correlation, Y, options, Target, kernelparam, lambda, alphaW, betaW, InvOKW, aaa, W);

% Cost function of twin Gaussian processes and its derivatives.
function [FYW, DFYW] = correlation(Y, Target, kernelparam, lambda, alphaW, betaW, InvOKW, aaa, W)

bbb = sum(Y.^2);
kvec = exp(-kernelparam*(aaa + bbb - 2*Target*Y));
kvecW = W*kvec;
InvOKWkvecW = InvOKW*kvecW;
ybetaW = (1+lambda) - kvecW'*InvOKWkvecW;
FYW = -2*alphaW'*kvecW - betaW*log(ybetaW);
TTTW = 2*(betaW/ybetaW*InvOKWkvecW - alphaW);
DFYW = 2*kernelparam.*((TTTW.*kvecW)'*Target - (TTTW'*kvecW)*Y')';

function Weight = LinearRegressor(Input, Target, Lambda)
%% Linear regression

[N, d] = size(Input);
BiasVec = ones(N,1);
Hessian = [BiasVec'*BiasVec BiasVec'*Input; Input'*BiasVec Input'*Input];
InputTarget = [sum(Target); Input'*Target];

if nargin < 3
    Lambda = 1e-5*mean(diag(Hessian));
else
    Lambda = Lambda*min(diag(Hessian));
end

Weight = (Hessian + Lambda*eye(d+1))\InputTarget;

