clear; clc; close all;
addpath(genpath('./thirdparty/minFunc_2012'));
% demonstrate TGP on S data
load('./thirdparty/data/SData');

Input = X(1:500); Target = t(1:500);
TestInput = X(501:1000); TestTarget = t(501:1000);

% figure;
% subplot(1,2,1); plot(Input,Target, '+');
% subplot(1,2,2); plot(TestInput,TestTarget, '+');

%%
minInput = min(Input); maxInput = max(Input);
maxTarget = max(Target); minTarget = min(Target);

minTstInput = min(TestInput); maxTstInput = max(TestInput);
maxTstTarget = max(TestTarget); minTstTarget = min(TestTarget);

k = 2;

InputExt = Input; TargetExt = Target;
TestInputExt = TestInput; TestTargetExt = TestTarget;

for i = 1:k-1
    InputExt = [InputExt - (maxTstInput-minTstInput) ; InputExt];
    TargetExt = [TargetExt - (maxTstTarget-minTstTarget) ; TargetExt];

    TestInputExt = [TestInputExt - (maxTstInput-minTstInput) ; TestInputExt];
    TestTargetExt = [TestTargetExt - (maxTstTarget-minTstTarget) ; TestTargetExt];
end

% subplot(1,2,1); plot(InputExt,TargetExt, '+');
% subplot(1,2,2); plot(TestInputExt,TestTargetExt, '+');


%%
Input = InputExt(1:2:end); Target = TargetExt(1:2:end);
TestInput = TestInputExt(1:2:end); TestTarget = TestTargetExt(1:2:end);

% figure;
% subplot(1,2,1); plot(Input,Target, '+');
% subplot(1,2,2); plot(TestInput,TestTarget, '+');

%% KL TGP
Param.kparam1 = 0.2; Param.kparam2 = 20;
Param.lambda = 1e-4; Param.SMAlpha = 0.5; Param.SMBeta = 0.5;

% Importance Weight Rescaling
% weights = ones(1,size(Input,1));
alphaparam = 0.5;
[PE1,weights,~]=RuLSIF(TestInput',Input',[],alphaparam,[],[],[],5);
W = diag(sqrt(max(0,weights)));

[InvIK, InvOK, IK, OK] = TGPTrain(Input, Target, Param);
tic
TGPPredKL = TGPTest(TestInput, Input, Target, Param, InvIK, InvOK);
toc
errKL = mean(abs(TGPPredKL(:)-TestTarget(:)));
disp(['Error of KLTGP is: ' num2str(errKL)]);

[InvIKW, InvOKW, IKW, OKW] = IWTGPTrain(Input, Target, Param, W);
tic
IWTGPPredKL = IWTGPTest(TestInput, Input, Target, Param, InvIKW, InvOKW, W);
toc
IWerrKL = mean(abs(IWTGPPredKL(:)-TestTarget(:)));
disp(['Error of IWKLTGP is: ' num2str(IWerrKL)]);

figure;
plot(TestInput,TestTarget,'.','Markersize',10);
hold on
[aaa,index] = sort(TestInput);
plot(aaa,TGPPredKL(index),'r+','Markersize',8);
hold on
plot(aaa,IWTGPPredKL(index),'g*','Markersize',8);
set(gca,'FontSize',16);
legend('Ground Truth','Prediction','Location','NorthWest');
xlabel('Input'); ylabel('Output');
title(['TGP (\gamma_{x} = 0.2, \gamma_{y} = 20 and Err = ',num2str(IWerrKL)]);

%% Sharma Mittal 
%% TGP SH
Param.kparam1 = 0.2;
Param.kparam2 = 20;
Param.lambda = 1e-4;
% Values alpha and beta could take (for cross validation
% alphas = [0.00000001, 0.000001,0.00001, 0.1, 0.2, 0.29, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.9999];
% betas = [ 7, 5, 3.5, 2.5, 2, 1.5, 0.999];
% betas = [ 1.5,2, 2.5,3,5,7];
Param.SMAlpha = 0.6;
Param.SMBeta  = 0.99;

% SMTGP
[InvIK, InvOK, IK, OK] = TGPTrain(Input, Target, Param);
IOKAlphaInv =  inv((1-Param.SMAlpha)* IK+ (Param.SMAlpha)*OK);
[SMTGPPred, SMTGPcert] = TGPSH4Test(TestInput, Input, Target, Param, InvIK, InvOK, IK, OK,TGPPredKL, IOKAlphaInv);
SMTGP_err = mean(abs(SMTGPPred(:)-TestTarget(:)));
certainties = mean(SMTGPcert);
disp(['Error of SMTGP is: ' num2str(SMTGP_err),', certainties of SMTGP is: ', num2str( mean(SMTGPcert)),', certainties std of SMTGP is: ' num2str( std(SMTGPcert))]);

% IW-SMTGP
[InvIKW, InvOKW, IKW, OKW] = IWTGPTrain(Input, Target, Param, W);
IOKWAlphaInv =  inv((1-Param.SMAlpha)* IKW + (Param.SMAlpha)*OKW);
[IWSMTGPPred, IWSMTGPcert] = IWSMTGPTest(TestInput, Input, Target, Param, W, InvIKW, InvOKW, IKW, OKW, TGPPredKL, IOKWAlphaInv);
IWSMTGP_err = mean(abs(IWSMTGPPred(:)-TestTarget(:)));
certainties = mean(IWSMTGPcert);
disp(['Error of IWSMTGP is: ' num2str(IWSMTGP_err),', certainties of SMTGP is: ', num2str( mean(IWSMTGPcert)),', certainties std of SMTGP is: ' num2str( std(IWSMTGPcert))]);

figure;
plot(TestInput,TestTarget,'.','Markersize',10);
hold on
[aaa,index] = sort(TestInput);
plot(aaa,SMTGPPred(index),'r+','Markersize',8);
hold on
plot(aaa,IWSMTGPPred(index),'g*','Markersize',8);
set(gca,'FontSize',16,'XLim',[-0.2 1.2],'YLim',[-0.2 1.2]);
axis tight
legend('Ground Truth','Prediction','Location','NorthWest');
xlabel('Input');
ylabel('Output');
title(['SMTGP (\gamma_{x} = 0.2, \gamma_{y} = 20 and Err = ', num2str(IWSMTGP_err), ')']);

%% Gaussian Process Regression
kparam = 20;
lambda = 1e-5;
K = EvalKernel(Input,Input,'rbf',kparam);
alpha = (K+lambda*eye(size(K)))\Target;
testK = EvalKernel(TestInput,Input,'rbf',kparam);
GPPred = testK*alpha;
disp(['Error of GPR is: ' num2str(mean(abs(GPPred(:)-TestTarget(:))))]);

GPRError_err =  mean(abs(GPPred(:)-TestTarget(:)));
figure;
plot(TestInput,TestTarget,'.','Markersize',10);
hold on
[aaa,index] = sort(TestInput);
plot(aaa,GPPred(index),'r+','Markersize',8);
set(gca,'FontSize',16,'XLim',[-0.2 1.2],'YLim',[-0.2 1.2]);
axis tight
legend('Ground Truth','Prediction','Location','NorthWest');
xlabel('Input');
ylabel('Output');
title(['GPR (\gamma_{x} = 0.2, \gamma_{y} = 20 and Err = ', num2str(GPRError_err), ')']);

%% Weighted K-Nearest Neighbour Regression with K = 1
WKNNPred = WKNNRegressor(TestInput, Input, Target, 1);
disp(['Error of WKNN (K=1) is: ' num2str(mean(abs(WKNNPred(:)-TestTarget(:))))]);
WKNNrror_err =  mean(abs(WKNNPred(:)-TestTarget(:)));

figure;
plot(TestInput,TestTarget,'.','Markersize',10);
hold on
[aaa,index] = sort(TestInput);
plot(aaa,WKNNPred(index),'r+','Markersize',8);
set(gca,'FontSize',16,'XLim',[-0.2 1.2],'YLim',[-0.2 1.2]);
axis tight
legend('Ground Truth','Prediction','Location','NorthWest');
xlabel('Input');
ylabel('Output');
title(['WKNN (\gamma_{x} = 0.2, \gamma_{y} = 20 and Err = ', num2str(WKNNrror_err), ')']);

%% Weighted K-Nearest Neighbor Regression with different K
K = 15;
WKNNPred = WKNNRegressor(TestInput,Input,Target,K);
%[WKNNError, WKNNErrorvec] = JointError(WKNNPred, TestTarget);
%disp(['WKNN: ' num2str(WKNNError)]);
disp(['Error of WKNN (K=15) is: ' num2str(mean(abs(WKNNPred(:)-TestTarget(:))))]);

%%
% %% Gaussian Process Regression
% kparam = 1;
% lambda = 1e-4;
% K = EvalKernel(Input,Input,'rbf',kparam);
% alpha = (K+lambda*eye(size(K)))\Target;
% testK = EvalKernel(TestInput,Input,'rbf',kparam);
% GPPred = testK*alpha;
% %[GPError, GPErrorvec] = JointError(GPPred, TestTarget);
% %disp(['GP: ' num2str(GPError)]);
% disp(['Error GPR is: ' num2str(mean(abs(GPPred(:)-TestTarget(:))))]);

% %% Hilbert-Schmidt Independent Criterion with K Nearest Neighbors
% Param.knn = 100;
% Param.kparam1 = 100;
% Param.kparam2 = 2*1e-5;
% Param.kparam1 =  2*1e-4;
% Param.kparam2 = 2*1e-4;
% HSICKNNPred = HSICKNN(TestInput, Input, Target, Param);
% %[HSICKNNError, HSICKNNErrorvec] = JointError(HSICKNNPred, TestTarget,1);
% disp(['Error of HSICKNN is: ' num2str(mean(abs(HSICKNNPred(:)-TestTarget(:))))]);
% %disp(['HSICKNN: ' num2str(HSICKNNError)]);
% 
% %% Kernel Target Alignment with K Nearest Neighbors
% Param.kparam1 =  2*1e-3;
% Param.kparam2 = 2*1e-3;
% KTAKNNPred = KTAKNN(TestInput, Input, Target, Param);
% %[KTAKNNError, KTAKNNErrorvec] = JointError(KTAKNNPred, TestTarget);
% %disp(['KTAKNN: ' num2str(KTAKNNError)]);
% disp(['Error of KTAKNN is: ' num2str(mean(abs(KTAKNNPred(:)-TestTarget(:))))]);
% 
