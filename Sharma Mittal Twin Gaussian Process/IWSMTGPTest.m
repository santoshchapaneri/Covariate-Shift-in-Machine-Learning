function [TGPTarget,relcert] = IWSMTGPTest(TestInput, Input, Target, Param, W, InvIKW, InvOKW, IKW, OKW, ManualInit, IOKWAlphaInv)
% Make the prediction using Twin Gaussian Processess

T = size(TestInput,1);
TGPTarget = zeros(T,size(Target,2));
% if(~exist('ManualInit','var'))
%     Weight = LinearRegressor(Input, Target);
% end
Weight = LinearRegressor(Input, Target);
%logDetIK = logdet(IK,'chol');
%logDetOK = logdet(OK, 'chol');

%CI = -Param.SMAlpha*(1 -Param.SMBeta)/ (2*(1-Param.SMAlpha) );
%CO = - (1 -Param.SMBeta)/2;
%CIO = - (1 -Param.SMBeta)/(2*(1-Param.SMAlpha));
% LCId  = zeros(size(Input,1)+1,1);
% LCId(size(Input,1)+1) = 1;

% IOKAlphaInv =  inv((1-Param.SMAlpha)* IK+ (Param.SMAlpha)*OK);
% Note by Santosh: This is training stuff, should not be done during
% testing

relcert = zeros(T,1);
for frame = 1:T
%     fprintf('%d\n',frame);
    % Initialize
    OneTestInput = TestInput(frame,:);
    EOneTestInput = [1 OneTestInput];
    InitTarget = (EOneTestInput*Weight)';
%     if(~exist('ManualInit','var'))
%         InitTarget = (EOneTestInput*Weight)';
%     else
%         InitTarget = ManualInit(frame,:)';
%     end
    % optimize
    IR = EvalKernel(Input,OneTestInput,'rbf',Param.kparam1);
    IRW = W*IR; % alpha
    cxW = (1+Param.lambda) - IRW'*InvIKW*IRW; % beta
    [Y, cert] = ComputeOutputIWSMTGP(InitTarget, Target, Param.kparam2, Param.lambda, InvOKW, IRW, Param.SMAlpha, Param.SMBeta, IOKWAlphaInv, cxW, W);
    
    TGPTarget(frame,:) = Y';
    relcert(frame) = cert;
end

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

%DFY = 2*kernelparam.*((TTT.*kvec)'*Target - (TTT'*kvec)*Y')';
