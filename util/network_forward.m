function [ out_map] = network_forward(net, im, fore_thr)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
    [height, width, ~] = size(im);
    input = prepare_img(im, false);
    out = net.forward({input});
    out_map = out{1};
    out_map = permute(out_map, [2,1,3]);
    out_map = imresize(out_map, [height, width]);
    out_map = double(out_map > fore_thr);
end

