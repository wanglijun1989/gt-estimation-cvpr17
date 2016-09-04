close all; clear; clc;
addpath(genpath('./external'))
addpath('/home/lijun/Research/Code/mexopencv/');
img_root = '/home/lijun/Research/DataSet/Saliency/ECSSD/ECSSD-Image/';

I = imread(sprintf('%s%s', img_root, '0146.jpg'));
[height, width, ~] = size(I);
model=load('models/forest/modelBsds'); 
model=model.model;
model.opts.nms=-1; model.opts.nThreads=4;
model.opts.multiscale=0; model.opts.sharpen=2;

%% set up opts for spDetect (see spDetect.m)
opts = spDetect;
opts.nThreads = 4;  % number of computation threads
opts.k = 512;       % controls scale of superpixels (big k -> big sp)
opts.alpha = .5;    % relative importance of regularity versus data terms
opts.beta = .9;     % relative importance of edge versus color terms
opts.merge = 0.;%0;     % set to small value to merge nearby superpixels at end
tic;
[E,~,~,segs]=edgesDetect(I,model);
[S,V_rgb, V_lab] = spDetect(I,E,opts);
toc;
[A, ~, ~]=spAffinities(S,E,segs,opts.nThreads);
figure(1);imshow(mat2gray(E)); figure(2);subplot(2, 2, 1); imshow(V_rgb);

sp_num = max(S(:));
sp_id = [];
V = cat(3, V_rgb, V_lab);
V = reshape(V, [], 6);
fea_sp = nan(8, sp_num);
for i = 1:sp_num
   sp_loc = find(S == i);
   if isempty(sp_loc)
       continue;
   end
   fea_sp(1:6, i) = V(sp_loc(1), :);
   [r, c] = ind2sub([height, width], sp_loc);
   fea_sp(7, i) = mean(r/height);
   fea_sp(8, i) = mean(c/width);
   sp_id = [sp_id; i];
end


1

