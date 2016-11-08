function [ out_map] = network_forward_sal(net, im)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
    [height, width, ~] = size(im);
    input = prepare_img(im, false);
    out_map = net.forward({input});
    out_map = out_map{1};
    out_map = permute(out_map, [2,1,3]);
    out_map = imresize(out_map, [height, width]);
% 
    a1 = net.blobs('upscore1').get_data;
    a1 = permute(a1, [2,1,3]);
    a2 = net.blobs('upscore2').get_data;
    a2 = permute(a2, [2,1,3]);
    a3 = net.blobs('upscore3').get_data;
    a3 = permute(a3, [2,1,3]);
    a4 = net.blobs('upscore4').get_data;
    a4 = permute(a4, [2,1,3]);
    figure(10);
    subplot(2,2,1);
    imagesc(a1(:,:,2));
    subplot(2,2,2);
    imagesc(a2(:,:,2));
    subplot(2,2,3);
    imagesc(a3(:,:,2));
    subplot(2,2,4);
    imagesc(a4(:,:,2));
end