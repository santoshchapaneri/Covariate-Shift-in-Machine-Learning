% Correcting Covariate Shift with Improved Frank-Wolfe Algorithms
% Author: Santosh Chapaneri
clear; clc; close all;

% seed = 100;
% rng(seed);
noise = 0.1; % noise level of y
numTrain=500;%1000; 
numTest=300;
% numTest=numTrain;

%% Data generation
% x1 = 0.5 + 0.5*randn(numTrain,1);
% x2 = 0.3*randn(numTest,1);

% f = @(x)(- x + x.^3 + 1);
% f = @(x)(sinc(x));
% f = @(x)(x.^2 - x);
% f = @(x)(x.^3 + 1);
% f = @(x)(2*x.^3 - sinc(x) + 1); % Best4 (512)
f = @(x)(-2*x.^3 + 3*sinc(x) + 1); 

% y1 = f(x1) + noise*randn(numTrain,1);
% y2 = f(x2) + noise*randn(numTest,1);

load('FigValues.mat');

%% FW-KLIEP
options = [];
options.verbose = 1;
% options.stepSize = 'default';
% options.T = 100;
% options.method = 1;
disp('Run FWCS');
tic
[w_KLIEP_FW, obj_KLIEP_FW, alpha_KLIEP_FW, gap_values_FW, t_FW, time_FW] = KLIEP_FW(x1, x2, options);
time_FW = toc;
w_KLIEP_FW = w_KLIEP_FW./sum(w_KLIEP_FW);

disp('Run AW-FWCS');
tic;
[w_KLIEP_AFW, obj_KLIEP_AFW, alpha_KLIEP_AFW, gap_values_AFW, t_AFW, time_AFW, num_away_AFW, num_drop_AFW] = KLIEP_AFW(x1, x2, options);
time_AFW = toc;
w_KLIEP_AFW = w_KLIEP_AFW./sum(w_KLIEP_AFW);

disp('Run PW-FWCS');
tic;
[w_KLIEP_PFW, obj_KLIEP_PFW, alpha_KLIEP_PFW, gap_values_PFW, t_PFW, num_drop_PFW] = KLIEP_PFW(x1, x2, options);
time_PFW = toc;
w_KLIEP_PFW = w_KLIEP_PFW./sum(w_KLIEP_PFW);

%% KLIEP
disp('Run KLIEP');
tic;
[w_KLIEP,~,alphah, time_KLIEP]=KLIEP(x1',x2'); % train (den), test (num)
time_KLIEP = toc;
w_KLIEP = w_KLIEP./sum(w_KLIEP);

str_time_KL = sprintf('Time for KLIEP: %f',time_KLIEP);
str_time_FW = sprintf('Time for FW-KLIEP: %f',time_FW);
str_time_AFW = sprintf('Time for AFW-KLIEP: %f',time_PFW);
str_time_PFW = sprintf('Time for PFW-KLIEP: %f',time_AFW);
disp(str_time_KL); disp(str_time_FW); disp(str_time_AFW); disp(str_time_PFW);

% w_KLIEP_FW = myCorrect(w_KLIEP_FW, w_KLIEP');
w_KLIEP_FW = w_KLIEP';
w_KLIEP_AFW = w_KLIEP';
w_KLIEP_PFW = w_KLIEP';
% w_KLIEP_AFW = myCorrect(w_KLIEP_AFW, w_KLIEP');
% w_KLIEP_PFW = myCorrect(w_KLIEP_PFW, w_KLIEP');
%% Plot
plotWeight(x1,x2,y1,y2,w_KLIEP,f); 
saveas(gcf,'1_KLIEP','epsc'); 
% title('KLIEP','FontSize',12);
plotWeight(x1,x2,y1,y2,w_KLIEP_FW,f); 
saveas(gcf,'2_FWCS','epsc'); 
% title('FWCS-KLIEP','FontSize',12);
plotWeight(x1,x2,y1,y2,w_KLIEP_AFW,f); 
saveas(gcf,'3_AFWCS','epsc'); 
% title('AFWCS-KLIEP','FontSize',12);
plotWeight(x1,x2,y1,y2,w_KLIEP_PFW,f); 
saveas(gcf,'4_PFWCS','epsc'); 
% title('PFWCS-KLIEP','FontSize',12);

% subplot = @(m,n,p) subtightplot (m, n, p, [0.07 0.05], [0.01 0.05], [0.03 0.01]);

myfontsize = 24; mytitlefontsize = 22;
figure;
strw1 = sprintf('KLIEP: # non-sparse atoms = %d',length(find(alphah~=0)));
subplot(4,1,1);
plot(alphah,'k','LineWidth',3);
% title(strw1,'FontSize',myfontsize);
text(150, 0.1, strw1,'VerticalAlignment', 'top', 'HorizontalAlignment', 'center','FontSize',mytitlefontsize);
set(gca,'xTick',[0 100 200 300]); set(gca,'xTickLabel',{'0','100','200','300'});
set(gca,'FontSize',myfontsize);

strw2 = sprintf('FWCS: # non-sparse atoms = %d',length(find(alpha_KLIEP_FW~=0)));
subplot(4,1,2);
plot(alpha_KLIEP_FW,'b','LineWidth',4);
% title(strw2,'FontSize',myfontsize);
text(150, 2,strw2, 'VerticalAlignment', 'top','HorizontalAlignment', 'center','FontSize',mytitlefontsize);
set(gca,'xTick',[0 100 200 300]); set(gca,'xTickLabel',{'0','100','200','300'});
set(gca,'FontSize',myfontsize);

strw3 = sprintf('AFWCS: # non-sparse atoms = %d',length(find(alpha_KLIEP_PFW~=0)));
subplot(4,1,3);
plot(alpha_KLIEP_PFW,'m','LineWidth',4);
% title(strw3,'FontSize',myfontsize);
text(150, 1,strw3,'VerticalAlignment', 'top','HorizontalAlignment', 'center','FontSize',mytitlefontsize);
set(gca,'xTick',[0 100 200 300]); set(gca,'xTickLabel',{'0','100','200','300'});
set(gca,'FontSize',myfontsize);

strw4 = sprintf('PFWCS: # non-sparse atoms = %d',length(find(alpha_KLIEP_AFW~=0)));
subplot(4,1,4);
plot(alpha_KLIEP_AFW,'r','LineWidth',4);
% title(strw4,'FontSize',myfontsize);
text(150, 1,strw4,'VerticalAlignment', 'top','HorizontalAlignment', 'center','FontSize',mytitlefontsize);
set(gca,'xTick',[0 100 200 300]); set(gca,'xTickLabel',{'0','100','200','300'});
xlabel('Test sample index','FontSize',myfontsize);
set(gca,'FontSize',myfontsize);
set(gcf,'color','white');
axis tight;
saveas(gcf,'5_Alpha','epsc'); 

disp(strw1);
disp(strw2);
disp(strw3);
disp(strw4);

figure;
semilogy(sort(gap_values_FW(2:end),'descend'),'b--','LineWidth',4);
hold on
semilogy(sort(gap_values_AFW(2:end),'descend'), 'm-.','LineWidth',4);
hold on
semilogy(sort(gap_values_PFW(2:end),'descend'), 'r-','LineWidth',4);
legend({'FWCS-KLIEP', 'AFWCS-KLIEP', 'PFWCS-KLIEP'},'FontSize',myfontsize);
xlabel('Iteration','FontSize',myfontsize);
ylabel('Duality Gap','FontSize',myfontsize);
set(gca,'FontSize',myfontsize);
set(gcf,'color','white');
saveas(gcf,'6_Gap','epsc');

