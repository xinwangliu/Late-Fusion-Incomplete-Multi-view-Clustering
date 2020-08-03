clear
clc
warning off;

path = './';
addpath(genpath(path));
addpath(genpath('./ClusteringEvaluation'));
dataName = 'proteinFold';
load([path,'dataset/',dataName,'_Kmatrix'],'KH','Y');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
numclass = length(unique(Y));
numker = size(KH,3);
num = size(KH,1);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%
KH = kcenter(KH);
KH = knorm(KH);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
qnorm = 2;
H = zeros(num,numclass,numker);
for ker = 1:numker
    H(:,:,ker) = mykernelkmeans(KH(:,:,ker), numclass);
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
epsionset = [0.1:0.1:0.9];
for ie =1:length(epsionset)
    res_mean = 0;
    for iter = 1:30
        load([path,'./generateAbsentMatrix/',dataName,'_missingRatio_',num2str(epsionset(ie)),...
            '_missingIndex_iter_',num2str(iter),'.mat'],'S');
        lambda = 2.^(-15:3:15);
        for lam = 1:length(lambda)
            H_star = IncompleteMultikernelLatefusionclusteringV1Hv(H,numclass,lambda(lam));
            res(lam,:) = myNMIACC(H_star,Y,numclass);
        end
        res_mean = res_mean + res;
    end
    res_mean = res_mean / 30;
end