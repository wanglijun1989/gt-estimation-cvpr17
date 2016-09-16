%% Add path and init caffe
caffe_root = '/home/lijun/Research/Code/caffe-blvc/';
addpath([caffe_root '/matlab/'],genpath('./external'), 'util/');
caffe.reset_all;
gpu_id = 0;
caffe.set_mode_gpu();
caffe.set_device(gpu_id);

model_weights = [caffe_root 'models/cvpr17-ILT/ILT-3_iter_12000.caffemodel'];
model_def = [caffe_root 'examples/cvpr17/deploy2.prototxt'];
phase = 'test';
net = caffe.Net(model_def, model_weights, phase);

%% Init SE model
model=load('models/forest/modelBsds'); 
model=model.model;
model.opts.nms=-1; model.opts.nThreads=4;
model.opts.multiscale=0; model.opts.sharpen=2;
%% set up opts for spDetect (see spDetect.m)
opts = spDetect;
opts.nThreads = 4;  % number of computation threads
opts.k = {512, 1024, 2048};       % controls scale of superpixels (big k -> big sp)
opts.alpha = .5;    % relative importance of regularity versus data terms
opts.beta = .9;     % relative importance of edge versus color terms
opts.merge = 0.;%0;     % set to small value to merge nearby superpixels at end
opts.num_scale = 3;
opts.scale_weight = [0.7, 0.2, 0.1];
assert(opts.num_scale == length(opts.k) && opts.num_scale == length(opts.scale_weight));
%% set up opts for CRF
crf_opt.fore_thr = 0.6;
crf_opt.fore_area_thr = 0.5;
crf_opt.fea_theta = [1e-2, 1e-2, 1e-2];
crf_opt.position_theta = [5e-3, 1e-2, 1e-2];
crf_opt.smooth_theta = [1e-4, 1e-4, 1e-4];
assert(opts.num_scale == length(crf_opt.fea_theta) && opts.num_scale == length(crf_opt.position_theta)...
    &&opts.num_scale == length(crf_opt.smooth_theta))

crf_opt.fea_theta2 = 1e-2;
crf_opt.position_theta2 = 5e-3;
crf_opt.smooth_theta2 = 1e-4;
%% set up opts for background cues
bgd_opt.reg = 50;
bgd_opt.margin_ratio = 0.1;
bgd_opt.background_cue_weight = 2;%%
visualize = false;