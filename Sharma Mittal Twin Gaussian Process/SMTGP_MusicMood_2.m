AMGFeats = csvread('AMG1608Feats.csv');

AMGFeats = AMGFeats(:,1:22); 
load('AMG1608AnnotatorsConsensusCIWM.mat');

AMGTarget = [AMG1608AnnotatorsConsensusCIWM.YArousal AMG1608AnnotatorsConsensusCIWM.YValence];

rp = randperm(1608);
numSamples = 804;
Input = AMGFeats(rp(1:numSamples),:);
Target = AMGTarget(rp(1:numSamples),:);
TestInput = AMGFeats(rp(numSamples+1:end),:);
TestTarget = AMGTarget(rp(numSamples+1:end),:);

[weight_PFW, obj_PFW, alpha_PFW] = KLIEP_PFW(Input,TestInput);
% [w_KLIEP,~,alpha_KLIEP]=KLIEP(Input',TestInput'); 
% [PE1,weights_RULSIF,~] = RuLSIF(TestInput',Input',[],0.5,[],[],[],5);

weights = weight_PFW;
W = diag(sqrt(max(0,weights)));

%% KL TGP
% [kparam1, kparam2] = KLTGPTuningParams(Input, Target, TestInput, TestTarget, 1);
Param.kparam1 = 1; Param.kparam2 = 0.05;
Param.lambda = 1e-4; 
[InvIK, InvOK, IK, OK] = TGPTrain(Input, Target, Param);
tic
TGPPredKL = TGPTest(TestInput, Input, Target, Param, InvIK, InvOK);
TGPTestTime = toc;
[TGPError, TGPErrorvec] = JointR2(TGPPredKL, TestTarget);
disp(['R2 of KLTGP is: ' num2str(TGPError)]);

% % IWTGP
% [InvIKW, InvOKW, IKW, OKW] = IWTGPTrain(Input, Target, Param, W);
% tic
% IWTGPPredKL = IWTGPTest(TestInput, Input, Target, Param, InvIKW, InvOKW, W);
% IWTGPTestTime = toc;
% [IWTGPError, IWTGPErrorvec] = JointError(IWTGPPredKL, TestTarget);
% disp(['Error of IWKLTGP is: ' num2str(IWTGPError)]);
% 
% % Direct IWTGP
% [kparam1, kparam2] = DTGPTuningParamsR2(Input, Target, TestInput, TestTarget, weights);

DWTGP_Param.kparam1 = 1; DWTGP_Param.kparam2 = 0.05;%1e-5; 50,100
DWTGP_Param.kparam3 = DWTGP_Param.kparam2;
DWTGP_Param.lambda = 1e-4;
DWTGP_Param.knn1 = min(length(Input),100); %M nearest neighbor
DWTGP_Param.wknnflag = 1; % Distance based weighting
DWTGP_Param.knn2 = 25; % K nearest neighbor
[DIWTGPKNNPred traintime testtime mu_all] = DWTGPKNN(TestInput, Input, Target, DWTGP_Param, weights);
DIWTGPTestTime = testtime;
[DIWTGPError, DWTGPErrorvec] = JointR2(DIWTGPKNNPred, TestTarget);
disp(['R2 of DIWKLTGP is: ' num2str(DIWTGPError)]);