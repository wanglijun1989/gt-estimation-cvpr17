close all; clear; clc;
init_test_gc;
rng(0);
%% Set data & resutls path
imgRoot='/home/lijun/Research/DataSet/Saliency/PASCAL-S/PASCAL-S-Image/';% test image path
% imgRoot='/home/lijun/Research/DataSet/Saliency/ECSSD/ECSSD-Image/';% test image path
% imgRoot='/home/lijun/Research/DataSet/Saliency/MSRA5000/MSRA5000-Image/';% test image path
% imgRoot = [data_path 'image/ILSVRC2013_DET_val/'];

% res_path='./crf_gmm_res_2048/';% the output path of the saliency map
% res_path = 'crf_gmm_res/MSRA5000/512/';
% res_path = 'crf_gmm_res/PASCAL-S/512-multi-scale-1/';
% res_path = 'crf_gmm_res/PASCAL-S/multi-scale-2/';
% res_path = 'crf_gmm_res/PASCAL-S/multi-scale-test/';
% res_path = 'crf_gmm_res/PASCAL-S/multi-scale-test-2/';
% res_path = 'crf_gmm_res/PASCAL-S/multi-scale-test-3/';
res_path = 'crf_gmm_res/PASCAL-S/multi-scale-test-6/';
% res_path = 'crf_gmm_res/ECSSD/512-back-prior-3/';
% res_path = 'crf_gmm_res/ECSSD/multi-scale-2/';
% res_path = 'crf_gmm_res/ECSSD/multi-scale-3/';
% res_path = 'crf_gmm_res/ECSSD/multi-scale-test-3/';
if ~isdir(res_path)
    mkdir(res_path);
end
imnames=dir([imgRoot '*' 'jpg']);
%% Main loop
for ii=155:length(imnames)
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

    %% Oversegemntation
    [superpixels, sp_num, affinity, feature] = OverSegment(im, model, opts);
    %% Compute superpixel init label and features (r,g,b,l,a,b,x,y)
    sp_info = ComputeSPixelFeature(superpixels, sp_num, gen_map, feature, background_cue, [height, width], crf_opt.fore_area_thr);
    [edge_appearance, edge_smooth, edge_affinity] = ComputeEdgeWeight(sp_info, affinity, crf_opt);
    edge_appearance(1:size(edge_affinity, 1)+1:end) = 0;
    edge_smooth(1:size(edge_affinity, 1)+1:end) = 0;
    edge_affinity(1:size(edge_affinity, 1)+1:end) = 0;
    boundary = DetectBoundarySP(superpixels, sp_num);
    %% Init CRF
    sp_feature = cell2mat(sp_info.fea');
    sp_position = cell2mat(sp_info.position');
    sp_init_label = cell2mat(sp_info.init_label');
    
    background_cue_sp = cell2mat(sp_info.background_cue');
    crf = CRF(255*[sp_feature; sp_position],sp_init_label, ...
        {edge_affinity, edge_appearance, edge_smooth}, [.1, 1, 0.5],...
        'boundary', boundary,'prior', background_cue_sp, 'prior_weight', 0, 'sp_num', sp_num);
    %% Show GMM labeling
        visualization(im, gen_map, superpixels, crf, opts.scale_weight, sp_num, visualize);
    %% CRF iteration
    try
        for iteration = 1:10
            crf.NextIter();
%             visualization(im, gen_map, superpixels, crf, opts.scale_weight, sp_num, visualize);
        end
    catch
        %         assert(0)
         crf = CRF(255*[sp_feature; sp_position],sp_init_label, ...
        {edge_affinity, edge_appearance, edge_smooth}, [.1, 1, 0.5],...
        'boundary', boundary,'prior', background_cue_sp, 'prior_weight', 0, 'sp_num', sp_num);
        crf.NextIter();
    end
    %% visualization and save results
    visualization(im, gen_map, superpixels, crf, opts.scale_weight, sp_num, visualize);
    res = GenerateMap(im, superpixels, crf, opts.scale_weight, sp_num);
    imwrite(res, [res_path imnames(ii).name(1:end-3) 'png']);
end