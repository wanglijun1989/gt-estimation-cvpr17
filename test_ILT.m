devkit_path = '/home/lijun/Research/DataSet/ILSVRC2014/ILSVRC2014_devkit/ILSVRC2014_devkit/';
data_path = '/home/lijun/Research/DataSet/ILSVRC2014/ILSVRC2014_DET/';
img_root = [data_path 'image/ILSVRC2013_DET_val/'];
% img_root = '/home/lijun/Research/DataSet/Saliency/ECSSD/ECSSD-Image/';
anno_root = [data_path 'BOX/ILSVRC2013_DET_bbox_val/'];
addpath([devkit_path 'evaluation/'],'data_prepare/');
load([devkit_path 'data/meta_det.mat']);
wnid2detid = Wnid2Detid(synsets);

caffe_root = '/home/lijun/Research/Code/caffe-blvc/';
addpath([caffe_root '/matlab/'])
caffe.reset_all;
gpu_id = 0;
caffe.set_mode_gpu();
caffe.set_device(gpu_id);
model_weights = [caffe_root 'models/cvpr17/ILT-exp_iter_1600.caffemodel'];
model_def = [caffe_root 'examples/cvpr17/deploy.prototxt'];
phase = 'test';
net = caffe.Net(model_def, model_weights, phase);
num_imgs = 10;
imgs = dir([img_root '*.JPEG']);
% imgs = dir([img_root '*.jpg']);
num_imgs = length(imgs);
for i = 8092:num_imgs
%     if mod(i, 100) == 0
        fprintf('Processing img: %04d / %04d\n', i, num_imgs);
%     end
    im = imread(sprintf('%s%s', img_root, imgs(i).name));
    % im = imread('img/i4.jpg');
    input = prepare_img(im, false);
    out = net.forward({input});
    out = out{1};
    
    
    
    im_name = imgs(i).name(1:end-5);
    ann = VOCreadxml([anno_root im_name '.xml']);
    try
        obj = zeros(length(ann.annotation.object), 1);
    catch
        continue;
    end
    for bb = 1 :length(ann.annotation.object)
        obj(bb) = wnid2detid(ann.annotation.object(bb).name);
    end
    obj = unique(obj);
   obj

    figure(1)
    plot(out,'-o');
    ylim([0,1]);
%     pause();

 
    feature_map = net.blobs('CAM_conv').get_data();
    w = net.params('CAM_pre',1).get_data();
    feature_map = reshape(feature_map, [], 1024);
    cam = feature_map * w;
    cam = reshape(cam, 31, 31, 200);
    cam = permute(cam, [2,1,3]);
    figure(2);
    subplot(1,2,1);
    imagesc(im);
    subplot(1,2,2);
    [~, id] = max(out);
    act_id = find(out > 0.5);
    if isempty(act_id);
        act_id = id;
    end
%     imagesc(cam(:,:,id));
    imagesc(sum(cam(:,:,act_id),3));
    title(sprintf('%d: %s', id, synsets(id).name))
 end
% imwrite(im, 'im.ppm');
caffe.reset_all();