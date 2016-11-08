%% Add path and init caffe
caffe_root = '/home/lijun/Research/Code/caffe-blvc/';
addpath([caffe_root '/matlab/'],genpath('./external'), 'util/');
caffe.reset_all;
gpu_id = 0;
caffe.set_mode_gpu();
caffe.set_device(gpu_id);
%  test_set = 'PASCAL-S';
test_set = 'ECSSD';
% test_set = 'THUS';
iteration = 30000;
model_version = 'cvpr17-sal4';
training_version = 'sal-1';
model_weights = [caffe_root 'models/' model_version '/' training_version '_iter_' num2str(iteration) '.caffemodel'];%3100
model_def = [caffe_root 'examples/' model_version '/deploy.prototxt'];
phase = 'test';
net = caffe.Net(model_def, model_weights, phase);
%% path
imgRoot=['/home/lijun/Research/DataSet/Saliency/' test_set '/' test_set '-Image/'];% test image path
imnames=dir([imgRoot '*' 'jpg']);
res_path = ['sal_res/' test_set '/' model_version '/' training_version '-' num2str(iteration) '/'];
if ~isdir(res_path)
    mkdir(res_path);
end
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
crf_opt.fore_thr = 0.8;
crf_opt.fore_area_thr = 0.5;
crf_opt.fea_theta = [1e-2];
crf_opt.position_theta = [3e-2];
crf_opt.smooth_theta = [1e-4];
assert(opts.num_scale == length(crf_opt.fea_theta) && opts.num_scale == length(crf_opt.position_theta)...
    &&opts.num_scale == length(crf_opt.smooth_theta))

crf_opt.fea_theta2 = 1e-4;
crf_opt.position_theta2 = 1e-4;
crf_opt.smooth_theta2 = 1e-4;
%% set up opts for background cues
bgd_opt.reg = 50;
bgd_opt.margin_ratio = 0.1;
bgd_opt.background_cue_weight = 2;%%
visualize = 1;
