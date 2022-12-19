% A toy demo for the paper
% Correcting Covariate Shift with the Frank-Wolfe Algorithm, IJCAI 2015
% Author: Junfeng Wen (junfeng.wen@ualberta.ca), University of Alberta
clear
clc

% seed = 100;
% rng(seed);
% t = 2^7; % number of training or test points
noise = 0.1; % noise level of y
t1=200;
t2=1000;

%% Data generation
x1 = randn(t1,1)/2 + 0.5;
x2 = 0.3*randn(t2,1);
% load('Demo_FWCS_x1_x2.mat');

f = @(x)(- x + x.^3 + 1);
% f = @(x)(sinc(x)+tan(x));
% x1 = randn(t,1)/2 + 0.5;
% y1 = f(x1) + noise*randn(t,1);
y1 = f(x1) + noise*randn(t1,1);
% x2 = 0.3*randn(t,1);
% y2 = f(x2) + noise*randn(t,1);
y2 = f(x2) + noise*randn(t2,1);

%% KMM
% options = [];
% options.verbose = 0;
% options.kernel = @gausskernel;
% options.kernel_param = median(pdist(x1));
% [w_KMM_FW, obj_KMM_FW] = KMM_FW(x1, x2, options);
% [w_KMM_FW_line, obj_KMM_FW_line] = KMM_FW_line(x1, x2, options);

%% KLIEP
[wh_x_de,~,alphah]=KLIEP(x1',x2'); % train (den), test (num)
wh_x_de = wh_x_de./sum(wh_x_de);

%% FW-KLIEP
options = [];
options.verbose = 1;
% options.stepSize = 'default';
% [w_KLIEP_FW, obj_KLIEP_FW, alpha_KLIEP_FW] = KLIEP_FW(x1, x2, options);
% w_KLIEP_FW = w_KLIEP_FW./sum(w_KLIEP_FW);

options.stepSize = 'lineSearch';
[w_KLIEP_FW_line, obj_KLIEP_FW_line, alpha_KLIEP_FW_line] = KLIEP_FW(x1, x2, options);
w_KLIEP_FW_line = w_KLIEP_FW_line./sum(w_KLIEP_FW_line);


%% Plot
weight = w_KLIEP_FW_line;
plotWeight(x1,x2,y1,y2,weight,f);
plotWeight(x1,x2,y1,y2,wh_x_de,f);
figure;subplot(2,1,1);plot(alpha_KLIEP_FW_line);title('FW-KLIEP');subplot(2,1,2);plot(alphah);title('KLIEP');set(gcf,'color','white');
