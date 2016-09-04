function [ Y ] = mvn( X )
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
Y = bsxfun(@minus, X, mean(X, 2));
Y = bsxfun(@rdivide, Y, std(Y,[], 2)+ 0.01);

end

