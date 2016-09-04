devkit_path = '/home/lijun/Research/DataSet/ILSVRC2014/ILSVRC2014_devkit/ILSVRC2014_devkit/';
% data_path = '/home/lijun/Research/DataSet/ILSVRC2014/ILSVRC2014_DET/';
% img_root = [data_path 'image/ILSVRC2013_DET_val/'];
% img_root = [data_path 'image/ILSVRC2014_DET_train/n01514859/'];
img_root = '/home/lijun/Research/DataSet/Saliency/ECSSD/ECSSD-Image/';
% anno_root = [data_path 'BOX/ILSVRC2013_DET_bbox_val/'];
addpath([devkit_path 'evaluation/'],'data_prepare/');
load([devkit_path 'data/meta_det.mat']);
wnid2detid = Wnid2Detid(synsets);

caffe_root = '/home/lijun/Research/Code/caffe-blvc/';
addpath([caffe_root '/matlab/'])
caffe.reset_all;
gpu_id = 0;
caffe.set_mode_gpu();
caffe.set_device(gpu_id);
model_weights = [caffe_root 'models/cvpr17-ILT/ILT-_iter_44000.caffemodel'];
model_def = [caffe_root 'examples/cvpr17/deploy2.prototxt'];
phase = 'test';
net = caffe.Net(model_def, model_weights, phase);
% imgs = dir([img_root '*.JPEG']);
imgs = dir([img_root '*.jpg']);
num_imgs = length(imgs);
for i = 55:num_imgs
    fprintf('Processing img: %04d / %04d\n', i, num_imgs);
    im = imread(sprintf('%s%s', img_root, imgs(i).name));
    resized_im = imresize(im, [256, 256]);
    input = prepare_img(im, false);
    out = net.forward({input});
%     weight = out{2};
    out = out{1};
    [~, id] = max(out);
    flag = true;
    figure(1);
    fg = subplot(1, 2, 1); imshow(im);
    subplot(1, 2, 2); plot(out, '-o');  title(sprintf('%d: %s', id, synsets(id).name))
    while flag
        bb = floor(getrect(fg));
        rectangle('Position', bb);
        
        input = im(bb(2):bb(2) + bb(4), bb(1) : bb(1)+bb(3), :);
        input = prepare_img(input, false);
        out = net.forward({input});
        out = out{1};
        [~, id] = max(out);
        figure(1);
        fg = subplot(1, 2, 1); imshow(im); rectangle('Position', bb);
        subplot(1, 2, 2); plot(out, '-o');  title(sprintf('%d: %s', id, synsets(id).name));
        [~, ~, sig] = ginput(1);
        if sig ~= 1
            flag = false;
        end
    end
end