clear;close all;clc;
addpath('/home/lijun/Research/Code/caffe-blvc/matlab/')
data_set = 'ECSSD';
data_path = ['/home/lijun/Research/DataSet/Saliency/' data_set '/' data_set '-Image/'];
res_path = ['./res/' data_set '/'];
if ~isdir(res_path)
    mkdir(res_path);
end
imgs = dir([data_path '*.jpg']);
gpu_id = 0;
caffe.set_mode_gpu();
caffe.set_device(gpu_id);
model_weights = 'model/vgg16CAM_train_iter_90000.caffemodel';
model_def = 'model/deploy_vgg16CAM.prototxt';
phase = 'test';
net = caffe.Net(model_def, model_weights, phase);
w = net.params('CAM_fc', 1).get_data();
num_imgs = length(imgs);
for i = 1:num_imgs
    if mod(i, 100) == 0
        fprintf('Processing img: %04d / %04d\n', i, num_imgs);
    end
    im = imread(sprintf('%s%s', data_path, imgs(i).name));
    % im = imread('img/i4.jpg');
    input = prepare_img(im, true);
    out = net.forward({input});
    out = out{1};
    [~, id] = max(out);
    map = net.blobs('CAM_conv').get_data();
    heat_map1 = reshape(map, [], 1024) * w(:, id);
    heat_map1 = reshape(heat_map1, 14, 14);
    heat_map1 = permute(heat_map1, [2,1,3]);
    heat_map = imresize(mat2gray(heat_map1), [size(im, 1), size(im, 2)]);
    % figure(1); subplot(1,2,1); imshow(im);
    % subplot(1,2,2); imshow(mat2gray(heat_map));
    prob_map = uint8(heat_map >= 1.5*mean(heat_map(:)))*255;
    prob_map(heat_map < 1.5*mean(heat_map(:)) & heat_map >= 0.7*mean(heat_map(:))) = 100;
    prob_map = repmat(prob_map, 1, 1, 3);
    res_name = [imgs(i).name(1:end-3) 'ppm'];
    figure(1); subplot(1,2,1);imagesc(im);
    subplot(1,2,2);imagesc(heat_map1)
%     imwrite(prob_map, sprintf('%s%s', res_path, res_name));
end
% imwrite(im, 'im.ppm');
caffe.reset_all();
