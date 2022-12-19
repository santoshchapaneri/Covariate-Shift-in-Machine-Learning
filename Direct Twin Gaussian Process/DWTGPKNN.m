function [TGPTarget traintime testtime mu_all] = DWTGPKNN(TestInput, Input, Target, Param,ratio)
% Make the prediction using Twin Gaussian Process with k nearest neighbors

T = size(TestInput,1);
TGPTarget = zeros(T,size(Target,2));

%Training dTGP
tic
Ka = EvalKernel(Input,Input, 'rbf',Param.kparam1);
La = EvalKernel(Target,Target, 'rbf', Param.kparam2);
traintime = toc;

tic
%Test
dist2a = EvalKernel(Input, TestInput,'dist2');
for frame = 1:T
    
    OneTestInput = TestInput(frame,:);
   
    %Find M nearest neighbors
    %In my experiments, M = 200 or 300 are enough.
    dist2 = dist2a(:,frame);
   
    [val,index] = sort(dist2);
    knnindex = index(1:Param.knn1);
    TInput = Input(knnindex,:);
    TTarget = Target(knnindex,:);
    
    K = Ka(knnindex,knnindex);
    L = La(knnindex,knnindex);
    
    %Importance Weight Rescaling
    W = diag(sqrt(max(0,ratio(knnindex))));

    %Estimating weight vector
    ui = EvalKernel(TInput,OneTestInput,'rbf',Param.kparam1);
    Wui = W*ui;
    
    u = W*((W*K*W + Param.lambda*eye(size(K)))\Wui);
    eta = 1 - ui'*u;
    
    ut = L*u;
    a = u'*ut;
    
    mu = (-eta + sqrt(eta^2 + 4*a*(1+Param.lambda)))/(2*a);
    
    mu_all(frame) = mu;
    
    tmp = min(1+Param.lambda, max(eps,mu*ut));%min(1,max(eps,mu*ut));
    
    %Compute distance
    switch Param.wknnflag
        case 1
            invu = 1./((-log(tmp)));
        case 2
            invu = ones(size(tmp));
    end

    %Regress an output with K nerest neighbors
    knn = min(Param.knn2,Param.knn1);
    [val,index] = sort(-invu);
    knnindex = index(1:knn);
        
    invu = invu(knnindex(1:knn));
    
    ninvu = invu/sum(invu);
    TGPTarget(frame,:) = ninvu'*TTarget(knnindex(1:knn),:);
end
testtime = toc;
