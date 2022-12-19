function wh = calcWeight(x,W,M,V)

k = size(V,3);
%Weight estimation
for i = 1:k
 temp = gaussPDF(x,M(:,i),V(:,:,i));
 if i == 1
     wh = temp*W(i);
 else
     wh = wh + temp*W(i);
 end
end