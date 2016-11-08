clear
clc
rng(0);
imagenet_root = '/home/lijun/Research/DataSet/ILSVRC2014/ILSVRC2014_DET/';
image_path = [imagenet_root 'image/ILSVRC2014_DET_train/'];
map_path = [imagenet_root 'sal_map_2_png/ILSVRC2014_DET_train/'];
if ~isdir(map_path)
    mkdir(map_path)
end
sub_dir = dir(image_path);
%% Add path and init caffe
caffe_root = '/home/lijun/Research/Code/caffe-blvc/';
addpath([caffe_root '/matlab/'],genpath('./external'), 'util/');
caffe.reset_all;
gpu_id = 1;
caffe.set_mode_gpu();
caffe.set_device(gpu_id);
%% specify machine id and model version
machine_id = 'cvpr17-sal-finetune';
major_model_version = num2str(2);
minor_model_version = '-0';
specify_machine;
iter_num = '50000';
%% init network
model_weights = [caffe_root 'models/' machine_path '/sal-finetune-' major_model_version minor_model_version '_iter_' iter_num '.caffemodel'];
model_def = [caffe_root 'models/' machine_path '/deploy-' major_model_version '.prototxt'];
phase = 'test';
net = caffe.Net(model_def, model_weights, phase);
%% 
%% Init SE model
model=load('models/forest/modelBsds'); 
model=model.model;
model.opts.nms=-1; model.opts.nThreads=4;
model.opts.multiscale=0; model.opts.sharpen=2;
%% set up opts for spDetect (see spDetect.m)
opts = spDetect;
opts.nThreads = 4;  % number of computation threads
opts.k = {512};       % controls scale of superpixels (big k -> big sp)
opts.alpha = .5;    % relative importance of regularity versus data terms
opts.beta = .9;     % relative importance of edge versus color terms
opts.merge = 0.;%0;     % set to small value to merge nearby superpixels at end
opts.num_scale = 1;
opts.scale_weight = [1];
assert(opts.num_scale == length(opts.k) && opts.num_scale == length(opts.scale_weight));
%% set up opts for CRF
crf_opt.fore_thr = 0.4;
crf_opt.fore_area_thr = 0.5;
crf_opt.fea_theta = [1e-2];
crf_opt.position_theta = [5e-3];
crf_opt.smooth_theta = [1e-4];
assert(opts.num_scale == length(crf_opt.fea_theta) && opts.num_scale == length(crf_opt.position_theta)...
    &&opts.num_scale == length(crf_opt.smooth_theta))
%% 
for dir_id = length(sub_dir) : -1 :455
    if strcmp(sub_dir(dir_id).name, '.') || strcmp(sub_dir(dir_id).name, '..')
        continue;
    end
    fprintf('Processing Directory: %d / %d \n', dir_id, length(sub_dir));
    cur_image_path = [image_path sub_dir(dir_id).name '/'];
    cur_map_path = [map_path sub_dir(dir_id).name '/'];
    if ~isdir(cur_map_path)
        mkdir(cur_map_path);
    end
    imgs = dir([cur_image_path '*JPEG']);
    if dir_id == 18
        start_im_id = 300;
    else 
        start_im_id = 1;
    end
    for im_id = start_im_id:length(imgs) % dir_id = 3 (fileid = 1), im_id = 8000:lengh(imgs), max_side<500
        if mod(im_id, 100) == 0
            fprintf('img: %d/%d \n', im_id, length(imgs));
        end
        try
        im = imread([cur_image_path imgs(im_id).name]);
        catch
            continue;
        end
        [ori_height, ori_width, ~] = size(im);
        max_side = max(ori_height, ori_width);
        if max_side > 500
            im = imresize(im, 500/max_side);    
        end
        
        [height, width, ch] = size(im);
        if ch ~= 3
            im = repmat(im, [1,1,3]);
        end
        %% forward pass
        gen_map = network_forward(net, im, crf_opt.fore_thr);
        
        %% Compute Background cues
        %     background_cue = BG(im, bgd_opt.reg, bgd_opt.margin_ratio);
        %     background_cue = (background_cue - min(background_cue(:))) / (max(background_cue(:)) - min(background_cue(:)));
        %     figure(1);subplot(1,2,1);imshow(background_cue)
        %     continue
        
        %% Oversegemntation
        [superpixels, sp_num, affinity, feature] = OverSegment(im, model, opts);
        %% Compute superpixel init label and features (r,g,b,l,a,b,x,y)
        sp_info = ComputeSPixelFeature(superpixels, sp_num, gen_map, feature, [height, width], crf_opt.fore_area_thr);
        [edge_appearance, edge_smooth, edge_affinity] = ComputeEdgeWeight(sp_info, affinity, crf_opt);
        edge_appearance(1:size(edge_affinity, 1)+1:end) = 0;
        edge_smooth(1:size(edge_affinity, 1)+1:end) = 0;
        edge_affinity(1:size(edge_affinity, 1)+1:end) = 0;
        boundary = DetectBoundarySP(superpixels, sp_num);
        %% Init CRF
        sp_feature = cell2mat(sp_info.fea');
        sp_position = cell2mat(sp_info.position');
        sp_init_label = cell2mat(sp_info.init_label');
        
        %     background_cue_sp = cell2mat(sp_info.background_cue');
        
        if sum(sp_init_label) <= 5
            if max_side > 500
              gen_map = imresize(gen_map, [ori_height, ori_width]);
            end
            res = uint8(gen_map>0.5);
            imwrite(res, [cur_map_path imgs(im_id).name]);
            continue;
        end
        try
        crf = CRF(255*[sp_feature; sp_position],sp_init_label, ...
            {edge_affinity, edge_appearance, edge_smooth}, [.1, 1, 0.5],...
            'boundary', boundary, 'sp_num', sp_num);
        catch
            if max_side > 500
              gen_map = imresize(gen_map, [ori_height, ori_width]);
            end
            res = uint8(gen_map>0.5);
            imwrite(res, [cur_map_path imgs(im_id).name]);
            continue;
        end
        %% Show GMM labeling
        %         visualization(im, gen_map, superpixels, crf, opts.scale_weight, sp_num, visualize);
        %% CRF iteration
        
        try
            for iteration = 1:5
                crf.NextIter();
                %             visualization(im, gen_map, superpixels, crf, opts.scale_weight, sp_num, visualize);
            end
        catch
            %         assert(0)
            crf = CRF(255*[sp_feature; sp_position],sp_init_label, ...
                {edge_affinity, edge_appearance, edge_smooth}, [.1, 1, 0.5],...
                'boundary', boundary, 'prior_weight', 0, 'sp_num', sp_num);
            %         crf.NextIter();
        end
        %% visualization and save results
        %     visualization(im, gen_map, superpixels, crf, opts.scale_weight, sp_num, visualize);
        res = GenerateMap(im, superpixels, crf, opts.scale_weight, sp_num);
        if max_side > 500
            res = imresize(res, [ori_height, ori_width]);
        end
        res = uint8(res>0.5);
        imwrite(res, [cur_map_path imgs(im_id).name(1:end-4) 'png']);
    end
end
caffe.reset_all;

