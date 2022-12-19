function prob = normpdf_mat(X,W,M,V)

[d,n] = size(X);
k = size(V,3);
s = (2*pi)^d;
prob = zeros(n,1);
for j = 1:n
    Q = 0;
    for i=1:k
        %if det(V(:,:,i)) > 10^(-10)
            P = W(i)/sqrt(s*(abs(det(V(:,:,i)))) + eps)*exp(-0.5*(X(:,j) - M(:,i))'*pinv(V(:,:,i))*(X(:,j) - M(:,i))); 
            Q = Q + P;
        %else
        %    P = 0;
        %end
    end
    prob(j) = Q;
end
    