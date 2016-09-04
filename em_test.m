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

num_sp = max(S(:));
fea_sp = nan(8, num_sp);
V = cat(3, V_rgb, V_lab);
V = reshape(V, [], 6);

for i = 1:num_sp
    sp_id = find(S == i);
    try
    fea_sp(1:6, i) = mean(V(sp_id, :));
    [r, c] = ind2sub([height, width], sp_id);
    fea_sp(7, i) = mean(r/height);
    fea_sp(8, i) = mean(c/width);
%     fea_sp(6, i) = 0.0000001 * (fea_sp(4, i)^2 + fea_sp(5, i)^2)^0.5; 
    catch
        continue;
    end
end
sp_id = ~isnan(fea_sp(1,:));
A = 0.1*A + 1 * eye(size(A));
fea_sp(1:6, sp_id) = fea_sp(1:6, sp_id) *  A(sp_id,sp_id);
fea_sp(1:6, sp_id) = bsxfun(@rdivide, fea_sp(1:6, sp_id), sum(A(sp_id, sp_id), 1)+ 0.0001);
em = cv.EM('Nclusters', 10, 'CovMatTyep', 'Spherical', 'MaxIters', 200);
% em = cv.EM('Nclusters', 50);
% em = cv.EM('Nclusters', 50, 'MaxIters', 200);
fea_sp(:, sp_id) = mvn(fea_sp(:, sp_id));
[~, label, ~] = em.train(fea_sp(:, sp_id)');
label = label + 1;

sp_label = zeros(1, num_sp);
sp_label(sp_id) = label;
sp_label = [0, sp_label];
S_label_map = sp_label(S+1);
S_label_map = uint8(mat2gray(S_label_map)*255);

RandStream.setGlobalStream(RandStream('mt19937ar', 'Seed', sum(clock)));
figure(2);subplot(2, 2, 2); imshow(uint8(S_label_map)); 
c_map = rand(256, 3); colormap(c_map)

edge_id = S_label_map == 0;
S_tmp = S_label_map;
S_label_map_wo_edge = S_label_map;
S_tmp(:, 1:end -1) = S_label_map(:, 2:end);
S_label_map_wo_edge(edge_id) = S_tmp(edge_id);

S_tmp = S_label_map_wo_edge;
S_tmp(:, 2:end) = S_label_map_wo_edge(:, 1:end-1);
edge_id = S_label_map_wo_edge == 0;
S_label_map_wo_edge(edge_id) = S_tmp(edge_id);

S_tmp = S_label_map_wo_edge;
S_tmp(1:end-1, :) = S_label_map_wo_edge(2:end, :);
edge_id = S_label_map_wo_edge == 0;
S_label_map_wo_edge(edge_id) = S_tmp(edge_id);

S_tmp = S_label_map_wo_edge;
S_tmp(2:end, :) = S_label_map_wo_edge(1:end-1, :);
edge_id = S_label_map_wo_edge == 0;
S_label_map_wo_edge(edge_id) = S_tmp(edge_id);

if ~isempty(find(S_label_map_wo_edge == 0, 1))
    1;
end

kernel = ones(3, 3);
kernel(2,2) = -8;
boundary = conv2(double(S_label_map_wo_edge), kernel, 'same');
boundary = boundary ~= 0;
I_c = I(:, :, 1);
I_c(boundary) = 0;
I_b(:, :, 1) =  I_c; 

I_c = I(:, :, 2);
I_c(boundary) = 255;
I_b(:, :, 2) =  I_c; 

I_c = I(:, :, 3);
I_c(boundary) = 255;
I_b(:, :, 3) =  I_c; 

figure(2);subplot(2, 2, 3); imshow(I_b);


