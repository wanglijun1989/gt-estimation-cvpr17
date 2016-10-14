%% Add path and init caffe
caffe_root = '/home/lijun/Research/Code/caffe-blvc/';
addpath([caffe_root '/matlab/'],genpath('./external'), 'util/');
caffe.reset_all;
gpu_id = 1;
caffe.set_mode_gpu();
caffe.set_device(gpu_id);

%% specify machine id and model version
machine_id = 'local';
major_model_version = num2str(1);
minor_model_version = '';
% iter_num = '22000';
% postfix = '65';
data_set = 'PASCAL-S';
specify_machine;
%% init network
model_weights = [caffe_root 'models/' machine_path '/ip-' major_model_version minor_model_version '_iter_' iter_num '.caffemodel'];
model_def = [caffe_root 'models/' machine_path '/deploy-' major_model_version '.prototxt'];
phase = 'test';
net = caffe.Net(model_def, model_weights, phase);
%% input output path

imgRoot=['/home/lijun/Research/DataSet/Saliency/' data_set '/' data_set '-Image/'];
res_path = ['crf_gmm_res/' data_set '/' machine_id '_v' major_model_version minor_model_version '/' iter_num '-' postfix '/'];
if ~isdir(res_path)
    mkdir(res_path);
end
imnames=dir([imgRoot '*' 'jpg']);
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
% crf_opt.fore_thr = 0.65;
crf_opt.fore_area_thr = 0.5;
crf_opt.fea_theta = [1e-2];
crf_opt.position_theta = [5e-3];
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
visualize = false;