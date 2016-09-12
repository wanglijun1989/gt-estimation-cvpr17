close all; clear; clc;
init_test_gc;
rng(0);
%% Set data & resutls path
% imgRoot='/home/lijun/Research/DataSet/Saliency/PASCAL-S/PASCAL-S-Image/';% test image path
imgRoot='/home/lijun/Research/DataSet/Saliency/ECSSD/ECSSD-Image/';% test image path
% imgRoot='/home/lijun/Research/DataSet/Saliency/MSRA5000/MSRA5000-Image/';% test image path
% imgRoot = [data_path 'image/ILSVRC2013_DET_val/'];

% res_path='./crf_gmm_res_2048/';% the output path of the saliency map
% res_path = 'crf_gmm_res/MSRA5000/512/';
% res_path = 'crf_gmm_res/PASCAL-S/512-back-prior-2/';
res_path = 'crf_gmm_res/ECSSD/512-back-prior-3/';
if ~isdir(res_path)
    mkdir(res_path);
end
imnames=dir([imgRoot '*' 'jpg']);
%% Main loop
for ii=1:length(imnames)
    fprintf('Processing Img:%d/%d\n', ii, length(imnames));
    %% read image
    im = imread(sprintf('%s%s', imgRoot, imnames(ii).name));
    [height, width, ch] = size(im);
    if ch ~= 3
        im = repmat(im, [1,1,3]);
    end
    %% forward pass
    gen_map = network_forward(net, im, crf_opt.fore_thr);

    %% Compute Background cues
    background_cue = BG(im, bgd_opt.reg, bgd_opt.margin_ratio);
    background_cue = (background_cue - min(background_cue(:))) / (max(background_cue(:)) - min(background_cue(:)));
    %     figure(1);subplot(1,2,1);imshow(background_cue)
    %     continue
    %% Oversegmentation
    [E,~,~,segs]=edgesDetect(im,model);
    [superpixels, V_rgb, V_lab] = spDetect(im,E,opts);
    superpixels = 1 + superpixels;
    [affinity,~,~]=spAffinities(superpixels,E,segs,opts.nThreads);
    superpixels = double(superpixels);
    sp_num=max(superpixels(:));
    assert(sp_num == length(unique(superpixels(:))))
    %% Compute superpixel init label and features (r,g,b,l,a,b,x,y)
    fea_sp = nan(6, sp_num);
    position = nan(2, sp_num);
    init_label = zeros(1, sp_num);
    V = cat(3, V_rgb, V_lab);
    V = reshape(V, [], 6);
    tmp = zeros(height, width);
    background_cue_sp = zeros(2, sp_num);
    for i = 1 : sp_num
        sp_loc = find(superpixels == i);
        fea_sp(:, i) = V(sp_loc(1), :);
        [r, c] = ind2sub([height, width], sp_loc);
        position(1, i) = mean(r/height);
        position(2, i) = mean(c/width);
        area = length(sp_loc);
        fore_area = sum(sum(gen_map(sp_loc)));
        background_cue_sp(2, i) = max(background_cue(sp_loc));
        background_cue_sp(1, i) = 1 - background_cue_sp(2, i);
        init_label(i) = double(fore_area / area > crf_opt.fore_area_thr);
    end
    %% compute edge weights and construct CRF
    
    edge_feature = ComputeSimilarity(fea_sp, crf_opt.fea_theta);
    edge_position = ComputeSimilarity(position, crf_opt.position_theta);
    edge_smooth = ComputeSimilarity(position, crf_opt.smooth_theta);
    edge_appearance = edge_feature .* edge_position;
    affinity(1:size(affinity, 1)+1:end) = 0;
    edge_appearance(1:size(affinity, 1)+1:end) = 0;
    edge_smooth(1:size(affinity, 1)+1:end) = 0;
    boundary = superpixels(1, :)';
    boundary = [boundary; superpixels(end, :)'];
    boundary = [boundary; superpixels(:, 1)];
    boundary = [boundary; superpixels(:, end)];
    boundary = unique(boundary);
    %     edge_appearance = bsxfun(@rdivide, edge_appearance, sum(edge_appearance, 1));
    %     edge_smooth = bsxfun(@rdivide, edge_smooth, sum(edge_smooth, 1));
    %% Init CRF
    crf = CRF(255*[fea_sp; position], init_label, ...
        {affinity, edge_appearance, edge_smooth}, [.1, 1, 0.5],...
        boundary, background_cue_sp, 0);
    %% Show GMM labeling
        visualization(im, gen_map, superpixels, crf, visualize, false);
    %% CRF iteration
    try
        for iteration = 1:10
            crf.NextIter();
            visualization(im, gen_map, superpixels, crf, visualize, false);
        end
    catch
        %         assert(0)
        crf = CRF(255*[fea_sp; position], init_label, {affinity, edge_appearance, edge_smooth}, [.1, 2, 0.5], boundary);
        crf.NextIter();
    end
    %% visualization and save results
    res = visualization(im, gen_map, superpixels, crf, visualize, true);
    imwrite(res, [res_path imnames(ii).name(1:end-3) 'png']);
end