function [Priors, Mu, Sigma,loglikall,tsall] = GMKLIEP(Data_te, Data_tr, Priors0, Mu0, Sigma0, delta)
%
% This function directly computes the ratio of two probability densities p_{te}(x)/p_{tr}(x)
% by using the Gaussian mixture models, where p_{te}(x) and p_{tr}(x) are
% training and test probability densities respectively.
%
% Inputs -----------------------------------------------------------------
%   o Data_te: D x N_{te} array representing N_{te} datapoints of D dimensions.
%   o Data_tr: D x N_{tr} array representing N_{tr} datapoints of D dimensions.
%   o Priors0: 1 x K array representing the initial prior probabilities
%              of the K GMM components.
%   o Mu0:     D x K array representing the initial centers of the K GMM
%              components.
%   o Sigma0:  D x D x K array representing the initial covariance matrices
%              of the K GMM components.
%   o delta :  Regulalization parameter for covariance matrices.
%
% Outputs ----------------------------------------------------------------
%   o Priors:  1 x K array representing the prior probabilities of the K GMM
%              components.
%   o Mu:      D x K array representing the centers of the K GMM components.
%   o Sigma:   D x D x K array representing the covariance matrices of the
%              K GMM components.

%% Criterion to stop the GMKLIEP iterative update
loglik_threshold = 0.0001;

%% Initialization of the parameters
[d, nte] = size(Data_te);
[d, ntr] = size(Data_tr);
nbStates = size(Sigma0,3);
loglik_old = -realmax;
nbStep = 0;

Mu = Mu0;
Sigma = Sigma0;
Priors = Priors0;

stcount = 1;
while 1
    
    stcount = stcount + 1;
    
    if stcount > 5000
        break;
    end
    
    %% E-step %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    for i=1:nbStates
        %Compute probability p(x|i)
        Pxi(:,i) = gaussPDF(Data_te, Mu(:,i), Sigma(:,:,i));
        Pxtr(:,i) = gaussPDF(Data_tr,Mu(:,i),Sigma(:,:,i));
    end
    
    
    Pix_tmp = repmat(Priors,[nte 1]).*Pxi;
    Pix = Pix_tmp ./ repmat(sum(Pix_tmp,2),[1 nbStates]);
    
    E = sum(Pix);
    B = sum(Pxtr);
    
    %count = 1;
    step = 0.1;%Step size of fixied-point algorithm
    %% M-step %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    for i=1:nbStates
        %Update the priors
        Priors(i) = E(i) / nte /B(i)* ntr;
        beta(i) = E(i)/nte - B(i)/ntr;
        
        %Update the means
        Mutmp = (Data_te*Pix(:,i)/nte - Data_tr*Pxtr(:,i)/ntr)/beta(i);
        MuNew(:,i) = (1-step)*Mu(:,i) + step*Mutmp;
        
        %Update the covariance matrices
        Data_tmp1 = Data_te - repmat(Mu(:,i),1,nte);
        Data_tmp1a = repmat(reshape(Data_tmp1,[d 1 nte]), [1 d 1]);
        Data_tmp1b = repmat(reshape(Data_tmp1,[1 d nte]), [d 1 1]);
        Data_tmp1c = repmat(reshape(Pix(:,i),[1 1 nte]), [d d 1]);
        
        Data_tmp2 = Data_tr - repmat(Mu(:,i),1,ntr);
        Data_tmp2a = repmat(reshape(Data_tmp2,[d 1 ntr]), [1 d 1]);
        Data_tmp2b = repmat(reshape(Data_tmp2,[1 d ntr]), [d 1 1]);
        Data_tmp2c = repmat(reshape(Pxtr(:,i),[1 1 ntr]), [d d 1]);
        
        Sigmatmp = sum(Data_tmp1a.*Data_tmp1b.*Data_tmp1c, 3)/nte - sum(Data_tmp2a.*Data_tmp2b.*Data_tmp2c, 3)/ntr;
        Sigmatmp = Sigmatmp/beta(i);
        ts = beta(i);
        
        Sigma(:,:,i) = (1-step)*Sigma(:,:,i) + step*Sigmatmp;
        
        %% Add a tiny variance to avoid numerical instability
        Sigma(:,:,i) = Sigma(:,:,i) + delta.*diag(ones(d,1));
        
        %If the numerical problem occured re-initialize the parameters
        nindex = isnan(Sigma(:,:,i));
        if sum(nindex) > 0
            MuNew(:,i) = Mu0(:,i);
            Sigma(:,:,i) = eye(d);
            Priors(i) = 0;
        end
        
    end
    %Update the means
    Mu = MuNew;
    %% Stopping criterion %%%%%%%%%%%%%%%%%%%%
    for i=1:nbStates
        %Compute probability p(x|i)
        Pxi(:,i) = gaussPDF(Data_te, Mu(:,i), Sigma(:,:,i));
        Pxtr(:,i) = gaussPDF(Data_tr,Mu(:,i),Sigma(:,:,i));
    end
    
    Pix_tmp = repmat(Priors,[nte 1]).*Pxi;
    Pix = Pix_tmp ./ repmat(sum(Pix_tmp,2),[1 nbStates]);
    
    F = Pxi*Priors';
    %keyboard
    F(find(F<eps)) = eps;
    loglik = mean(log(F));
    
    
    if abs((loglik/loglik_old)-1) < loglik_threshold %| loglik < loglik_old
        break;
    end
    
    loglik_old = loglik;
    loglikall(stcount) = loglik;
    tsall(stcount) =ts;
    %count = count + 1;
    nbStep = nbStep+1;
end


%% Add a tiny variance to avoid numerical instability
for i=1:nbStates
    Sigma(:,:,i) = Sigma(:,:,i) + delta.*diag(ones(d,1));
end

