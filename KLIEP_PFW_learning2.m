% Solving KLIEP with Pairwise Aways Steps Frank-Wolfe algorithm 
% Learning function
function [weight, obj, alpha, gap_values, t, num_drop] = KLIEP_PFW_learning2(Xtr, Xte, options)

method = options.method;
tol = options.tol;
T = options.T;
[ntr, d] = size(Xtr);
nte = size(Xte, 1);
Ktr_te = gausskernel(Xtr,Xte,options.sigma);
Kte_te = gausskernel(Xte,Xte,options.sigma);
alpha = zeros(nte, 1);
mu_t = zeros(nte,1); 
k_sum_tr = sum(Ktr_te, 1) + 1e-30; % add small value for stability
bounds = ntr./k_sum_tr';
[~, ib] = min(bounds);
alpha(ib) = bounds(ib); % initialization
mu_t(ib) = 1;

track = []; 
S = find(mu_t>0); % active set of indices

gap_values = [];
num_drop = 0; % counting drop steps (max stepsize for away step)

t = 0; obj = -inf; diff = inf;
while (method == 1 && t < T) ||...
        (method == 2 && abs(diff) > tol) ||...
        (method == 3 && options.target-obj > tol)
    if t == 0
        Kalpha = Kte_te*alpha;
    else
%         Kalpha = Kalpha + stepSize*(Kte_te(:,oldIdx)*bounds(oldIdx) - Kalpha);
        Kalpha = (1 - stepSize).*Kalpha +...
            Kte_te(:,oldIdx)*(stepSize*bounds(oldIdx));
%         Kalpha = (1 - stepSize).*Kalpha +...
%             (stepSize*bounds(oldIdx)).*Kte_te(:,oldIdx);
    end
    g = Kte_te'*(1./Kalpha); % gradient of alpha, all positive
    if any(isnan(g)) || any(isinf(g))
        weight = ones(ntr,1); obj = -inf;
        if options.verbose
            warning('Sigma too small.');
        end
        return;
    end

    % FW
    [~, id_FW] = max(g.*bounds); % maximizing obj
    e = zeros(nte,1); e(id_FW) = 1;
    d_FW = (bounds.*e - alpha); % FW
    % duality gap:
    gap = d_FW' * g;
    gap_values(t+1) = gap;

    %     if gap < tol
    %         fprintf('end of FW: reach small duality gap (gap=%g)\n', gap);
    %         break
    %     end 

    % Away
    [~, tmp] = min(g(S).*bounds(S)); % minimizing obj
    id_A = S(tmp);
    e = zeros(nte,1); e(id_A) = 1;
    d_A = alpha - bounds.*e; % Away
    mu_max = mu_t(id_A);

    % Construct pairwise direction (between towards and away)
    d = d_FW - d_A;
    max_step = mu_max;
    
%     max_rho = 1; idx = id_FW;
%     stepSize_FW = armijo(max_rho,alpha,d_FW,@myfun,@mygradfun);
%     max_rho = mu_max; idx = id_A;
%     stepSize_A = armijo(max_rho,alpha,d_A,@myfun,@mygradfun);
    
%     idx = [id_FW id_A];
%     idx = id_A;
%     alphaK = Kalpha';
%     stepSize = lineSearch;
    stepSize = 1/(t+2); % or linesearch
    stepSize = max(0, min(stepSize, max_step ));
    
    % Update active set
    % Away part
    if abs(stepSize - max_step) < 10*eps
        % drop step:
        num_drop = num_drop+1;
        mu_t(id_A) = 0;
        S(S == id_A) = []; % remove from active set
%         oldIdx = [id_FW]; 
    else
        mu_t(id_A) = mu_t(id_A) - stepSize;
%         oldIdx = [id_FW id_A]; % include both
    end
    
%     oldIdx = idx;
    % Towards part
    mu_t(id_FW) = mu_t(id_FW) + stepSize;
    S = find(mu_t > 0); 
    oldIdx = S;
    
    % exceptional case: stepsize of 1, this collapses the active set!
    if stepSize > 1-eps
        S = id_FW;
    end
    
    alpha = alpha + stepSize * d;
    
%     oldIdx = [id_FW id_A]; % include both

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

    if 0 % verbose
        fprintf('it = %d;  obj = %g; gap=%g; stepsize=%g; active = %d\n', t, obj, gap, stepSize, length(S));
    end

    t = t + 1;
end
% if options.verbose > 1
%     plot(track);
% end
weight = Ktr_te*alpha;

% function middle = lineSearch
%     % check the gradient on [0,1] instead of the objective
%     % the gradient is decreasing in (0,1) with g(0)>0
%     % try to find the step size rho s.t. g(rho) = 0
%     % if g(1)>=0, then 1 is the step size
%     % otherwise g(1)<0, use binary search
%     lower = 0; upper = 1; prec = inf; middle = 1;
%     
% %     sK = Kte_te(idx,:).*bounds(idx);
%     sK = bounds(idx)'*Kte_te(idx,:);
%     % alphaK can be updated more efficiently in the main loop
%     grad = sum((sK-alphaK)./((1-middle).*alphaK+middle.*sK));
%     if grad >= 0
%         return;
%     end
%     while prec > options.tol
%         middle = (lower + upper)/2;
%         grad = sum((sK-alphaK)./((1-middle).*alphaK+middle.*sK));
%         if grad > 0
%             lower = middle;
%         else
%             upper = middle;
%         end
%         prec = upper - lower;
%     end
% end

% function funval = myfun(alpha)
%     funval = sum(log(Kte_te*alpha))/nte;
% end
% 
% function gradval = mygradfun(alpha, rho)
%     if t == 0
%         Kalpha = Kte_te*alpha;
%     else
%         Kalpha = (1 - rho).*Kalpha +...
%             (rho*bounds(idx)).*Kte_te(:,idx);
%     end
%     gradval = Kte_te'*(1./Kalpha);
% end
% 
% function [rho_armijo] = armijo(rho,x,d,myfun,mygradfun,gamma,delta)
%     if (nargin<6)
%         delta = 0.5;
%         gamma = 1e-4;
%     elseif (nargin==6)
%         delta = 0.5;
%     end
%     
%     j = 1;
%     while (j>0)
%         x_new = x+rho.*d;
%         lhs = myfun(x_new);
%         rhs = myfun(x) + gamma*rho*mygradfun(x,rho)'*d;
%         if lhs <= rhs
%             j = 0;
%             rho_armijo = rho;
%         else
%             rho = rho*delta;
%         end    
%     end
% end

end
