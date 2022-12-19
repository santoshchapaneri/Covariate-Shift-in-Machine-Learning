function [optmix optdelta] = GMKLIEP_LCV(Data_te, Data_tr)
%
% This function directly computes the ratio of two probability densities p_{te}(x)/p_{tr}(x)
% by using the Gaussian mixture models, where p_{te}(x) and p_{tr}(x) are
% training and test probability densities respectively.
%
% Inputs -----------------------------------------------------------------
%   o Data_te: D x N_{te} array representing N_{te} datapoints of D dimensions.
%   o Data_tr: D x N_{tr} array representing N_{tr} datapoints of D dimensions.
%
% Outputs ----------------------------------------------------------------
%   o optmix:  Number of mixtures chosen by LCV
%   o optdelta:  Optimum regularization parameter chosen by LCV

xtest = Data_te;
xtrain = Data_tr;

[d,ntest] = size(Data_te);
[d,ntrain] = size(Data_tr);

fold=5;
cv_index=randperm(ntest);
cv_split=floor([0:ntest-1]*fold./ntest)+1;

%Model candidates
mixlist = [1 2 3];
deltalist = [10^(-5) 10^(-10)];
LCV_all = zeros(length(mixlist),length(deltalist));

for nn = 1:length(mixlist)
    k = mixlist(nn);
    
    for xx = 1:length(deltalist)

        delta = deltalist(xx);
        x = [xtest';xtrain'];
        [Wini,Mini,Vini] = Init_EM(x,k); %kmeans;
        
        for i=1:fold
            cv_train_index=cv_index(cv_split~=i);
            cv_test_index=cv_index(cv_split==i);
                    
            [W, M, V,loglik,ts] = GMKLIEP(xtest(:,cv_train_index), xtrain, Wini, Mini, Vini,delta);
            
             for i=1:k
                 Pxi(:,i) = gaussPDF(xtest(:,cv_test_index), M(:,i), V(:,:,i));
             end
             prob = Pxi*W';
            %[mix, options, errlog] = gmkliep(mixini, xtest(cv_train_index,:), xtrain, options);
            %prob = gmmprob(mix, xtest(cv_test_index,:));
            LCV_all(nn,xx) = LCV_all(nn,xx) + mean(log(prob + realmin))/fold;
        end
        
    end
end
nanindex = isnan(LCV_all);
LCV_all(nanindex) = -inf;

[val,ind1] = max(LCV_all);
[val,ind2] = max(val);

optmix = mixlist(ind1(ind2));

optdelta = deltalist(ind2);



