%% init caffe
close all; clear; clc;
init_test_sal;
rng(0);


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
    sal_map = network_forward_sal(net, im);
    sal_map = sal_map(:,:,2);
    sal_map = imresize(sal_map, [height, width]);
    imwrite(sal_map, [res_path imnames(ii).name(1:end-3) 'png']);
    figure(1);
    subplot(1,2,1); imshow(im);
    subplot(1,2,2); imshow(mat2gray(sal_map));
    continue
     %% Oversegemntation
    [superpixels, sp_num, affinity, feature] = OverSegment(im, model, opts);
      %% Compute Background cues
    background_cue = BG(im, bgd_opt.reg, bgd_opt.margin_ratio);
    background_cue = (background_cue - min(background_cue(:))) / (max(background_cue(:)) - min(background_cue(:)));
    %% Compute superpixel init label and features (r,g,b,l,a,b,x,y)
    sp_info = ComputeSPixelFeature(superpixels, sp_num, sal_map, feature, background_cue, [height, width], crf_opt.fore_area_thr);
    [edge_appearance, edge_smooth, edge_affinity] = ComputeEdgeWeight(sp_info, affinity, crf_opt);
    edge_appearance(1:size(edge_affinity, 1)+1:end) = 0;
    edge_smooth(1:size(edge_affinity, 1)+1:end) = 0;
    edge_appearance = edge_appearance .* double(edge_appearance > 0.1);
    edge_smooth = edge_smooth .* double(edge_smooth > 0.1);
    edge_affinity(1:size(edge_affinity, 1)+1:end) = 0;
    boundary = DetectBoundarySP(superpixels, sp_num);
    %% Init CRF
    sp_feature = cell2mat(sp_info.fea');
    sp_position = cell2mat(sp_info.position');
    sp_init_label = cell2mat(sp_info.init_label');
    
    background_cue_sp = cell2mat(sp_info.background_cue');
    try
        crf = CRF(255*[sp_feature; sp_position],sp_init_label, ...
            {edge_affinity, edge_appearance, edge_smooth}, [.1, 1, 1],...
            'boundary', boundary,'prior', background_cue_sp, 'prior_weight', 0, 'sp_num', sp_num);
    catch
        imwrite(mat2gray(sal_map), [res_path imnames(ii).name(1:end-3) 'png']);
        fprintf('Warning: init map is not accurate!\n');
        continue;
    end
    %% Show GMM labeling
        visualization(im, sal_map, superpixels, crf, opts.scale_weight, sp_num, visualize);
    %% CRF iteration
    try
        for iteration = 1:10
            crf.NextIter();
            visualization(im, sal_map, superpixels, crf, opts.scale_weight, sp_num, visualize);
        end
    catch
       
        %         assert(0)
         crf = CRF(255*[sp_feature; sp_position],sp_init_label, ...
        {edge_affinity, edge_appearance, edge_smooth}, [.01, 4, 1.5],...
        'boundary', boundary,'prior', background_cue_sp, 'prior_weight', 0, 'sp_num', sp_num);
    fprintf('Warning2: init map is not accurate!\n');
%         crf.NextIter();
    end
    %% visualization and save results
    visualization(im, sal_map, superpixels, crf, opts.scale_weight, sp_num, visualize);
    res = GenerateMap(im, superpixels, crf, opts.scale_weight, sp_num);
    imwrite(res, [res_path imnames(ii).name(1:end-3) 'png']);
end
caffe.reset_all;
    