function Y =elm2(P,W1,W2,B1,B2,LW,TF1,TF2,TYPE)
% ELM2 Create and Train a Extreme Learning Machine
% Syntax
% Y =elm2(P,W1,W2,B1,B2,LW,TF1,TF2,TYPE)
% Description
% Input
%P   - Input Matrix of Training Set  (R*Q)
% W1  - Input Weight Matrix (N*R)
% W2  - Input Weight Matrix of Neurons In the Second Hidden Layer £¨L*N)
% B1   - Bias Matrix of Neurons In the First Hidden Layer (N*1)
% B2   - Bias Matrix of Neurons In the Second Hidden Layer (L*1)
% LW  - Layer Weight Matrix (L*S)
% TYPE - Regression (0,default) or Classification (1)
% Output
% Y   - Simulate Output Matrix (S*Q)
% Example
% [W1,W2,B1,B2,LW,TF1,TF2,TYPE]=elm1(P,T,7,8,sig,sin,0)
% Y = elm2(P,W1,W2,B1,B2,LW,TF1,TF2,TYPE)
% Classification
% [W1,W2,B1,B2,LW,TF1,TF2,TYPE]=elm1(P,T,7,8,sig,sin,1)
% Y = elm2(P,W1,W2,B1,B2,LW,TF1,TF2,TYPE)
if nargin < 9
    error('ELM:Arguments','Not enough input arguments.');
end
% Calculate the Layer Output Matrix H
Q = size(P,2);
BiasMatrix1 = repmat(B1,1,Q);
BiasMatrix2 = repmat(B2,1,Q);
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
Y = (H2' * LW)';
