function [ w ] = LSE( v , r )
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here

w = 1/r*log(1/length(v)*sum(exp(r*v)));
end

