devkit_path = '/home/lijun/Research/DataSet/ILSVRC2014/ILSVRC2014_devkit/ILSVRC2014_devkit/';
data_path = '/home/lijun/Research/DataSet/ILSVRC2014/ILSVRC2014_DET/';
img_root = [data_path 'image/ILSVRC2013_DET_val/'];
% img_root = [data_path 'image/ILSVRC2014_DET_train/n01514859/'];
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
server_id = 11;
model_weights = [caffe_root 'models/server' num2str(server_id) '/ip-1_iter_62000.caffemodel'];
% model_weights = ['/home/lijun/ip-1_iter_3000.caffemodel'];
% model_weights = '/home/lijun/Research/Code/CVPR17/model/vgg16CAM_train_iter_90000.caffemodel';
model_def = [caffe_root 'models/server' num2str(server_id) '/deploy.prototxt'];
phase = 'test';
net = caffe.Net(model_def, model_weights, phase);
imgs = dir([img_root '*.JPEG']);
% imgs = dir([img_root '*.jpg']);
num_imgs = length(imgs);
for i = 1:num_imgs
%     if mod(i, 100) == 0
        fprintf('Processing img: %04d / %04d\n', i, num_imgs);
%     end
    im = imread(sprintf('%s%s', img_root, imgs(i).name));
    resized_im = imresize(im, [256, 256]);
    % im = imread('img/i4.jpg');
    input = prepare_img(im, false);
    out = net.forward({input});
    pre = out{1}(:);
    [~, id] = max(pre);
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
   
conv5_score = net.blobs('conv5_3_score').get_data();
conv5_score =  imresize(permute(conv5_score, [2,1,3]), [256, 256]);
figure(1);
subplot(2,2,1); imagesc(resized_im);
subplot(2,2,2);
plot(pre, '-o');
ylim([0,1]);
title(sprintf('%d: %s', id, synsets(id).name));
subplot(2,2,3); imagesc(conv5_score);
subplot(2,2,4); imagesc(bsxfun(@times, im2double(resized_im), double(conv5_score > 0.6)));

 x=net.blobs('conv_o_5_3').get_data;
x=permute(x,[2,1,3]);
y = sum(x, 3);
figure(4);imagesc(imresize(y,[256,256]))
% 
 x=net.blobs('conv6').get_data;
x=permute(x,[2,1,3]);
y = imresize(sum(x, 3),[256,256]);
% y = net.blobs('pool6').get_data;
% figure(5);plot(y(:), '-o')
figure(5);
subplot(1,2,1);imagesc(y);
subplot(1,2,2);imagesc(bsxfun(@times, im2double(resized_im), double(mat2gray(y) > 0.8)));

 end
% imwrite(im, 'im.ppm');
caffe.reset_all();
