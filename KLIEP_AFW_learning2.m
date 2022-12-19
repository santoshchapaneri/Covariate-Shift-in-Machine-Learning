% Solving KLIEP with Aways Steps Frank-Wolfe algorithm 
% Learning function
function [weight, obj, alpha, gap_values, t, num_away, num_drop] = KLIEP_AFW_learning2(Xtr, Xte, options)

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
num_away = 0;
num_drop = 0; % counting drop steps (max stepsize for away step)

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

    % FW
    [~, id_FW] = max(g.*bounds); % maximizing obj
    e = zeros(nte,1); e(id_FW) = 1;
    d_FW = (bounds.*e - alpha); % FW
    % duality gap:
    gap = d_FW' * g; % gap = - d_FW' * g;
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

    % construct direction (between towards and away):
    if (d_FW' * g >= d_A' * g)
        is_aw = false;
        d = d_FW; 
        max_step = 1;
        idx = id_FW;
    else % away step
        is_aw = true;
        num_away = num_away+1;
        d = d_A;
        max_step = mu_max / (1 - mu_max);
        idx = id_A;
    end

%     alphaK = Kalpha';
%     stepSize = lineSearch;
    stepSize = 1/(t+2); % or linesearch
    stepSize = max(0, min(stepSize, max_step ));

    if is_aw
      % fprintf('  AWAY step from index %d\n',id_A);
      % Away step:
      mu_t = (1+stepSize)*mu_t; % note that inactive should stay at 0;
      mu_t(id_A) = mu_t(id_A) - stepSize;
      if abs(stepSize - max_step) < 10*eps
          % drop step:
          num_drop = num_drop+1;
          mu_t(id_A) = 0;
          S(S == id_A) = []; % remove from active set
      else
          mu_t(id_A) = mu_t(id_A) - stepSize;
      end
    else
      % FW step:
      mu_t = (1-stepSize)*mu_t;

      mu_t(id_FW) = mu_t(id_FW) + stepSize;

      S = find(mu_t > 0); 

      % exceptional case: stepsize of 1, this collapses the active set!
      if stepSize > 1-eps;
          S = id_FW;
      end
    end

    alpha = alpha + stepSize * d;
    
    oldIdx = idx;
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

function middle = lineSearch
    % check the gradient on [0,1] instead of the objective
    % the gradient is decreasing in (0,1) with g(0)>0
    % try to find the step size rho s.t. g(rho) = 0
    % if g(1)>=0, then 1 is the step size
    % otherwise g(1)<0, use binary search
    lower = 0; upper = 1; prec = inf; middle = 1;
    sK = Kte_te(idx,:).*bounds(idx);
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
