function HP = updateHPabsentClusteringV1(WP,Hstar,lambda,HP00)

num = size(HP00,1);
k = size(HP00,2);
numker = size(HP00,3);
HP = zeros(num,k,numker);
for p = 1:numker
    Vp = Hstar*WP(:,:,p)' + lambda*HP00(:,:,p);
    [Up,Sp,Vp] = svd(Vp,'econ');
    HP(:,:,p) = Up*Vp';
end