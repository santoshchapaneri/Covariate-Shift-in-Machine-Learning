% Direct TGP program
%
% (c) Makoto Yamada, Yahoo Labs
%     makotoy@yahoo-inc.com
% This code was created while I was in CMU.

clear all
close all;
cpath = pwd;

seed = 0;
rand('state',seed);
randn('state',seed);

mutr=0.5;
sigmatr=0.3^2;
mute1=0.8;
mute2=0.2;
sigmate=0.3^2;

sigmay=0.05;

ntr=2000;
nte=500;


acc = 10000;
slope = 0.3;
ytrain = 1:acc;%linspace(0,1,10000);
xtrain = ytrain/acc + slope*sin(2*pi*ytrain/acc);

range_tr = [0.0001 1]*acc;
range_te1 = floor([0.85 1]*acc);
range_te2 = floor([0.0001 0.15]*acc);

yind_tr = randi(range_tr,1,ntr);
yind_te = [randi(range_tr,1,nte)];% randi(range_te2,1,nte/2)];

ytr = ytrain(yind_tr)/acc;
yte = ytrain(yind_te)/acc;

xtr = xtrain(yind_tr) + 0.0*randn(1,length(yind_tr));
xte = xtrain(yind_te) + 0.0*randn(1,length(yind_te));

ytr = ytr + 0.05*randn(1,length(yind_tr));
yte = yte + 0.05*randn(1,length(yind_te));


ydisp=linspace(-1.1,1.1,200);
xdisp=ydisp + slope*sin(2*pi*ydisp);

%Importance Weight
% wh_xtr = ones(1,size(xtr,2));
alphaparam = 0.5;
[PE1,wh_xtr,wh_xte]=RuLSIF(xte,xtr,xte,alphaparam,[],[],[],5);

 
Param.kparam1 = 0.5;
Param.kparam2 = 10;
Param.kparam3 = Param.kparam2;
Param.lambda = 10^(-5);
Param.knn1 = min(ntr,200); %M nearest neighbor
Param.wknnflag = 1; %Distance based weighting
Param.knn2 = 25; %K nearest neighbor

%Direct IWTGP
[WTGPKNNPred traintime testtime mu_all] = DWTGPKNN(xte', xtr', ytr',Param,wh_xtr);

err1 = mean((yte - WTGPKNNPred').^2);

figure(1);clf;hold on
set(gca,'FontSize',30)
h1 = plot(xtr(1:3:end),ytr(1:3:end),'ro', 'LineWidth',2,'Markersize',12);
plot(xdisp,ydisp,'k--','LineWidth',3)

xlabel('x')
ylabel('y')
hl = legend([h1],'$(x,y)$','Location','NorthWest');
set(hl, 'interpreter', 'latex');
axis([-0.05 1.05 -0.15 1.15]);
hold off
print('-depsc','data_toy')

figure(2);clf;hold on
set(gca,'FontSize',30)
plot(xdisp,ydisp,'k--','LineWidth',3)
plot(xte,WTGPKNNPred','r+','LineWidth',2,'Color',[0 200 0]/255,'Markersize',10);
xlabel('x')
ylabel('y')
sstr = sprintf('Error: %0.5g, TestTime:%0.3g',err1,testtime);
title(sstr);
axis([-0.05 1.05 -0.15 1.15]);
legend('True', 'Estimated','Location', 'NorthWest');
hold off
print('-depsc','DTGP_toy')


