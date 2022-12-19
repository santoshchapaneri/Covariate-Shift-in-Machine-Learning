% Solving KLIEP with Frank-Wolfe algorithm 
% Learning function
% Last modified: Jan. 2015
% Author: Junfeng Wen (junfeng.wen@ualberta.ca), University of Alberta
function [weight, obj, alpha, gap_values, t] = KLIEP_FW_learning2(Xtr, Xte, options)

method = options.method;
tol = options.tol;
T = options.T;
[ntr, d] = size(Xtr);
nte = size(Xte, 1);
Ktr_te = gausskernel(Xtr,Xte,options.sigma);
Kte_te = gausskernel(Xte,Xte,options.sigma);
alpha = zeros(nte, 1);
k_sum_tr = sum(Ktr_te, 1) + 1e-30; % add small value for stability
bounds = ntr./k_sum_tr';
[~, ib] = min(bounds);
alpha(ib) = bounds(ib); % initialization

track = [];
gap_values = [];

t = 0; obj = -inf; diff = inf;
while (method == 1 && t < T) ||...
        (method == 2 && abs(diff) > tol) ||...
        (method == 3 && options.target-obj > tol)
    if t == 0
        Kalpha = Kte_te*alpha;
    else
        Kalpha = (1 - stepSize).*Kalpha +...
            (stepSize*bounds(oldIdx)).*Kte_te(:,oldIdx);
    end
    g = Kte_te'*(1./Kalpha); % gradient of alpha, all positive
    if any(isnan(g)) || any(isinf(g))
        weight = ones(ntr,1); obj = -inf;
        if options.verbose
            warning('Sigma too small.');
        end
        return;
    end
    [~, id_FW] = max(g.*bounds); % maximizing obj
    
    e = zeros(nte,1); e(id_FW) = 1;
    d_FW = (bounds.*e - alpha); % FW
    % duality gap:
    gap = d_FW' * g; 
    gap_values(t+1) = gap;
    
    stepSize = 2/(t+2);
%     alphaK = Kalpha';
%     stepSize = lineSearch; % line search
    
    alpha = alpha + stepSize * d_FW;
%     alpha = (1 - stepSize).*alpha;
%     alpha(idx) = alpha(idx) + stepSize*bounds(idx);
    oldIdx = id_FW;
    if method == 2
        oldObj = obj;
        obj = sum(log(Kte_te*alpha))/nte;
        diff = obj - oldObj; % maximization
    elseif method == 3
        obj = sum(log(Kte_te*alpha))/nte;
    end
    if options.verbose > 1
        track = [track, sum(log(Kte_te*alpha))/nte];
    end
    t = t + 1;
end
if options.verbose > 1
    plot(track);
end
weight = Ktr_te*alpha;

function middle = lineSearch
    % check the gradient on [0,1] instead of the objective
    % the gradient is decreasing in (0,1) with g(0)>0
    % try to find the step size rho s.t. g(rho) = 0
    % if g(1)>=0, then 1 is the step size
    % otherwise g(1)<0, use binary search
    lower = 0; upper = 1; prec = inf; middle = 1;
    sK = Kte_te(id_FW,:).*bounds(id_FW);
    % alphaK can be updated more efficiently in the main loop
    grad = sum((sK-alphaK)./((1-middle).*alphaK+middle.*sK));
    if grad >= 0
        return;
    end
    while prec > options.tol
        middle = (lower + upper)/2;
        grad = sum((sK-alphaK)./((1-middle).*alphaK+middle.*sK));
        if grad > 0
            lower = middle;
        else
            upper = middle;
        end
        prec = upper - lower;
    end
end

end
