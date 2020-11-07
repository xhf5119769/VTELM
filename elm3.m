function [W1,W2,B1,B2,LW,TF1,TF2,TYPE]=elm1(P,T,N1,N2,TF1,TF2,TYPE)
% ELM1 Create and Train a Extreme Learning Machine
% Syntax
% [IW,B,LW,TF,TYPE] = elmtrain(P,T,N,TF,TYPE)
% Description
% Input
%P   - Input Matrix of Training Set  (R*Q)
% T   - Output Matrix of Training Set (S*Q)
% N1   - Number of first Hidden Neurons (default = Q)
% N2   - Number of second Hidden Neurons (default = Q)
% TF1  - Transfer Function:
%       'sig' for Sigmoidal function (default)
%       'sin' for Sine function
%       'hardlim' for Hardlim function
% TF2  - Transfer Function:
%       'sig' for Sigmoidal function (default)
%       'sin' for Sine function
%       'hardlim' for Hardlim function
% TYPE - Regression (0,default) or Classification (1)
% Output
% W1  - Input Weight Matrix (N*R)
% W2  - Input Weight Matrix of Neurons In the Second Hidden Layer （L*N)
% B1   - Bias Matrix of Neurons In the First Hidden Layer (N*1)
% B2   - Bias Matrix of Neurons In the Second Hidden Layer (L*1)
% LW  - Layer Weight Matrix (L*S)
% Example
% [W1,W2,B1,B2,LW,TF1,TF2,TYPE]=elm1(P,T,7,8,sig,sin,0)
% Y = elm2(P,W1,W2,B1,B2,LW,TF1,TF2,TYPE)
% Classification
% [W1,W2,B1,B2,LW,TF1,TF2,TYPE]=elm1(P,T,7,8,sig,sin,1)
% Y = elm2(P,W1,W2,B1,B2,LW,TF1,TF2,TYPE)
if nargin < 2
    error('ELM:Arguments','Not enough input arguments.');
end
if nargin < 4
    N1 = size(P,2);
    N2 = size(P,2);
end
if nargin < 6
    TF1 = 'sig';
    TF 2= 'sig';
end
if nargin < 7
    TYPE = 0;
end
if size(P,2) ~= size(T,2)
    error('ELM:Arguments','The columns of P and T must be same.');
end
[R,Q] = size(P);
if TYPE  == 1
    T  = ind2vec(T);
end
[S,Q] = size(T);
% Randomly Generate the Input Weight Matrix
W1 = rand(N1,R) * 2 - 1;
W2 = rand(N2,N1) * 2 - 1;
% Randomly Generate the Bias Matrix
B1 = rand(N1,1);
B2 = rand(N2,1);
BiasMatrix1 = repmat(B1,1,Q);
BiasMatrix2 = repmat(B2,1,Q);
% Calculate the Layer Output Matrix H
tempH1 = W1 * P + BiasMatrix1;
switch TF1
    case 'sig'
        H1 = 1 ./ (1 + exp(-tempH1));
    case 'sin'
        H1 = sin(tempH1);
    case 'hardlim'
        H1 = hardlim(tempH1);
     case 'dsig' 
        H1 = (1-exp(tempH1)) ./ (1 + exp(tempH1)); 
            
end
tempH2 = W2 * H1 + BiasMatrix2;
switch TF2
    case 'sig'
        H2 = 1 ./ (1 + exp(-tempH2));
    case 'sin'
        H2 = sin(tempH2);
    case 'hardlim'
        H2 = hardlim(tempH2);
           case 'dsig' 
        H2 = (1-exp(tempH2)) ./ (1 + exp(tempH2)); 
end
%LW = pinv(H2') * T';
%% N2为隐含层节点数，Q为训练集或者测试集样本个数
if(N2<Q)
     LW = (pinv(rand*eye(Q,Q)+H2'*H2)*H2')'* T';  % N*S   如果隐含层结点个数小于训练集或者测试集的样本个数时
elseif(N2>=Q)
     LW = (H2'*pinv(rand*eye(N2,N2)+H2*H2'))'* T';  % N*S   如果隐含层结点个数大于训练集或者测试集的样本个数时
% elseif(N==Q)
%      LW = pinv(H') * T';   % pinv(B)求的是矩阵B的Moore-Penrose逆
end