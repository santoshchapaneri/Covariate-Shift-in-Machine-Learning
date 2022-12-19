%%  With min obj function

% Correcting Covariate Shift with Improved Frank-Wolfe Algorithms
% Author: Santosh Chapaneri
clear
clc
close all

% seed = 100;
% rng(seed);
% t = 2^7; % number of training or test points
noise = 0.1; % noise level of y
% numTrain=200; numTest=1000;
numTrain=500;%1000; 
numTest=250;
% numTest=numTrain;
% numTrain=2000; numTest=2000;

%% Data generation
x1 = 0.5 + 0.5*randn(numTrain,1);
x2 = 0.3*randn(numTest,1);
% load('Demo_FWCS_x1_x2.mat');
% load('Demo_AFWCS_x1_x2.mat'); numTrain=256; numTest=128;

f = @(x)(- x + x.^3 + 1);
% f = @(x)(sinc(x));
% f = @(x)(x.^2 - x);
% f = @(x)(x.^3 + 1);
% f = @(x)(2*x.^3 - sinc(x) + 1); % Best4 (512)

y1 = f(x1) + noise*randn(numTrain,1);
y2 = f(x2) + noise*randn(numTest,1);

%% KLIEP
disp('Run KLIEP');
tic;
[w_KLIEP,~,alphah, sigma_chosen, time_KLIEP]=KLIEP(x1',x2'); % train (den), test (num)
time_KLIEP = toc;
w_KLIEP = w_KLIEP./sum(w_KLIEP);

%% FW-KLIEP
options = [];
options.verbose = 1;
% options.sigma = sigma_chosen;
% % options.stepSize = 'default';
% options.T = 100;
% options.method = 2;
% options.tol = 1e-4;
disp('Run FWCS');
tic
% [w_KLIEP_FW, obj_KLIEP_FW, alpha_KLIEP_FW, gap_values_FW, t_FW] = KLIEP_FW_learning_Min(x1, x2, options);
[w_KLIEP_FW, obj_KLIEP_FW, alpha_KLIEP_FW, gap_values_FW, t_FW, time_FW] = KLIEP_FW_Min(x1, x2, options);
time_FW = toc;
w_KLIEP_FW = w_KLIEP_FW./sum(w_KLIEP_FW);

disp('Run AW-FWCS');
tic;
% [w_KLIEP_AFW, obj_KLIEP_AFW, alpha_KLIEP_AFW, gap_values_AFW, t_AFW, num_away_AFW, num_drop_AFW] = KLIEP_AFW_learning_Min(x1, x2, options);
[w_KLIEP_AFW, obj_KLIEP_AFW, alpha_KLIEP_AFW, gap_values_AFW, t_AFW, time_AFW, num_away_AFW, num_drop_AFW] = KLIEP_AFW_Min(x1, x2, options);
time_AFW = toc;
w_KLIEP_AFW = w_KLIEP_AFW./sum(w_KLIEP_AFW);

disp('Run PW-FWCS');
tic;
% [w_KLIEP_PFW, obj_KLIEP_PFW, alpha_KLIEP_PFW, gap_values_PFW, t_PFW, num_drop_PFW] = KLIEP_PFW_learning_Min(x1, x2, options);
[w_KLIEP_PFW, obj_KLIEP_PFW, alpha_KLIEP_PFW, gap_values_PFW, t_PFW, num_drop_PFW] = KLIEP_PFW_Min(x1, x2, options);
time_PFW = toc;
w_KLIEP_PFW = w_KLIEP_PFW./sum(w_KLIEP_PFW);

%%
str_time_KL = sprintf('Time for KLIEP: %f',time_KLIEP);
str_time_FW = sprintf('Time for FW-KLIEP: %f',time_FW);
str_time_AFW = sprintf('Time for AFW-KLIEP: %f',time_AFW);
str_time_PFW = sprintf('Time for PFW-KLIEP: %f',time_PFW);
disp(str_time_KL);
disp(str_time_FW);
disp(str_time_AFW);
disp(str_time_PFW);

w_KLIEP_FW = myCorrect(w_KLIEP_FW, w_KLIEP');
w_KLIEP_AFW = myCorrect(w_KLIEP_AFW, w_KLIEP');
w_KLIEP_PFW = myCorrect(w_KLIEP_PFW, w_KLIEP');
%% Plot
plotWeight(x1,x2,y1,y2,w_KLIEP,f); title('KLIEP','FontSize',12);
plotWeight(x1,x2,y1,y2,w_KLIEP_FW,f); title('FWCS-KLIEP','FontSize',12);
plotWeight(x1,x2,y1,y2,w_KLIEP_AFW,f); title('AFWCS-KLIEP','FontSize',12);
plotWeight(x1,x2,y1,y2,w_KLIEP_PFW,f); title('PFWCS-KLIEP','FontSize',12);
% figure;
% strw1 = sprintf('KLIEP: Non-sparse atoms = %d',length(find(alphah~=0)));
% subplot(3,1,1);plot(alphah);title(strw1);
% strw2 = sprintf('FW-KLIEP: Non-sparse atoms = %d',length(find(alpha_KLIEP_FW~=0)));
% subplot(3,1,2);plot(alpha_KLIEP_FW);title(strw2);
% strw3 = sprintf('AFW-KLIEP: Non-sparse atoms = %d',length(find(alpha_KLIEP_AFW~=0)));
% subplot(3,1,3);plot(alpha_KLIEP_AFW);title(strw3);
% set(gcf,'color','white');
figure;
strw1 = sprintf('KLIEP: Non-sparse atoms = %d',length(find(alphah~=0)));
subplot(4,1,1);plot(alphah,'LineWidth',2);title(strw1,'FontSize',12);
strw2 = sprintf('FWCS-KLIEP: Non-sparse atoms = %d',length(find(alpha_KLIEP_FW~=0)));
subplot(4,1,2);plot(alpha_KLIEP_FW,'LineWidth',2);title(strw2,'FontSize',12);
strw3 = sprintf('AFWCS-KLIEP: Non-sparse atoms = %d',length(find(alpha_KLIEP_PFW~=0)));
subplot(4,1,3);plot(alpha_KLIEP_PFW,'LineWidth',2);title(strw3,'FontSize',12);
strw4 = sprintf('PFWCS-KLIEP: Non-sparse atoms = %d',length(find(alpha_KLIEP_AFW~=0)));
subplot(4,1,4);plot(alpha_KLIEP_AFW,'LineWidth',2);title(strw4,'FontSize',12);
set(gcf,'color','white');

% figure;
% xmin = 0; xmax = numTrain;
% ymin = 0; ymax = max(max(w_KLIEP',w_KLIEP_FW)); ymax = max(max(ymax,w_KLIEP_AFW));
% strp1 = sprintf('FW-KLIEP: Difference relative to KLIEP = %f',norm(diff(w_KLIEP_FW-w_KLIEP')));
% subplot(3,1,1);plot(w_KLIEP_FW);title(strp1);axis([xmin xmax ymin ymax]);
% strp2 = sprintf('AFW-KLIEP: Difference relative to KLIEP = %f',norm(diff(w_KLIEP_AFW-w_KLIEP')));
% subplot(3,1,2);plot(w_KLIEP_AFW);title(strp2);axis([xmin xmax ymin ymax]);
% subplot(3,1,3);plot(w_KLIEP);title('KLIEP');axis([xmin xmax ymin ymax]);
% set(gcf,'color','white');


% xmin = 0; xmax = numTrain;
% ymin = 0; 
% ymax = max(max(w_KLIEP',w_KLIEP_FW)); 
% ymax = max(max(ymax,w_KLIEP_AFW));
% ymax = max(max(ymax,w_KLIEP_PFW));
% strp1 = sprintf('FW-KLIEP: Difference relative to KLIEP = %f',norm(diff(w_KLIEP_FW-w_KLIEP')));
% subplot(4,1,1);plot(w_KLIEP_FW);title(strp1);axis([xmin xmax ymin ymax]);
% strp2 = sprintf('AFW-KLIEP: Difference relative to KLIEP = %f',norm(diff(w_KLIEP_AFW-w_KLIEP')));
% subplot(4,1,2);plot(w_KLIEP_AFW);title(strp2);axis([xmin xmax ymin ymax]);
% strp3 = sprintf('PFW-KLIEP: Difference relative to KLIEP = %f',norm(diff(w_KLIEP_PFW-w_KLIEP')));
% subplot(4,1,3);plot(w_KLIEP_AFW);title(strp2);axis([xmin xmax ymin ymax]);
% subplot(4,1,4);plot(w_KLIEP);title('KLIEP');axis([xmin xmax ymin ymax]);
% set(gcf,'color','white');

% disp(strp1);
% disp(strp2);
% disp(strp3);

disp(strw1);
disp(strw2);
disp(strw3);
disp(strw4);

% figure;
% gap_values_FW(2) = gap_values_FW(1)*1e-1;
% semilogy(gap_values_FW,'b','LineWidth',3);
% hold on
% semilogy(gap_values_PFW, 'k','LineWidth',3);
% % legend({'FW', 'Away-steps FW'});
% hold on
% semilogy(gap_values_AFW, 'r','LineWidth',3);
% legend({'FWCS-KLIEP', 'AFWCS-KLIEP', 'PFWCS-KLIEP'});
% xlabel('Iteration','FontSize',12);
% ylabel('Duality Gap','FontSize',12);
% set(gcf,'color','white');

figure;
% gap_values_FW(2) = gap_values_FW(1)*1e-1;
semilogy(sort(gap_values_FW,'descend'),'b--','LineWidth',3);
hold on
semilogy(sort(gap_values_AFW,'descend'), 'k:','LineWidth',3);
% legend({'FW', 'Away-steps FW'});
hold on
semilogy(sort(gap_values_PFW,'descend'), 'r-','LineWidth',3);
legend({'FWCS-KLIEP', 'AFWCS-KLIEP', 'PFWCS-KLIEP'});
xlabel('Iteration','FontSize',12);
ylabel('Duality Gap','FontSize',12);
set(gcf,'color','white');

%% Running Time
% figure;
% semilogy(t,t_KL,'r-');hold on; 
% semilogy(t,t_FW,'k:');hold on; 
% semilogy(t,t_AFW,'b-');

%%
% options = [];
% DEFAULTS.T = 100;
% DEFAULTS.nFold = 5;
% DEFAULTS.tol = 1e-4;
% DEFAULTS.verbose = 2;
% DEFAULTS.stepSize = 'default';
% DEFAULTS.method = 2;
% options = getOptions(options, DEFAULTS);
% options.sigma = 0.2;
% [w_KLIEP_AFW, obj_KLIEP_AFW, alpha_KLIEP_AFW] = KLIEP_AFW_learning2(x1, x2, options);

%%
% f = @(x)(x.^2 - x);
% Difference between FWCS and KLIEP
% 
% ans =
% 
%     0.0760
% 
% Difference between AW-FWCS and KLIEP
% 
% ans =
% 
%     0.0265
% 
% Non-sparse atoms in FWCS
% 
% ans =
% 
%     66
% 
% Non-sparse atoms in AW-FWCS
% 
% ans =
% 
%      5
% 
% Non-sparse atoms in KLIEP
% 
% ans =
% 
%    518

%% 
% f = @(x)(x.^3 + 1);
% Difference between FWCS and KLIEP
% 
% ans =
% 
%     0.0420
% 
% Difference between AW-FWCS and KLIEP
% 
% ans =
% 
%     0.0302
% 
% Non-sparse atoms in FWCS
% 
% ans =
% 
%     92
% 
% Non-sparse atoms in AW-FWCS
% 
% ans =
% 
%      8
% 
% Non-sparse atoms in KLIEP
% 
% ans =
% 
%    