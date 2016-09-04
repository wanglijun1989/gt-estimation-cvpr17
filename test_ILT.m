devkit_path = '/home/lijun/Research/DataSet/ILSVRC2014/ILSVRC2014_devkit/ILSVRC2014_devkit/';
data_path = '/home/lijun/Research/DataSet/ILSVRC2014/ILSVRC2014_DET/';
% img_root = [data_path 'image/ILSVRC2013_DET_val/'];
% img_root = [data_path 'image/ILSVRC2014_DET_train/n01514859/'];
img_root = '/home/lijun/Research/DataSet/Saliency/ECSSD/ECSSD-Image/';
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
model_weights = [caffe_root 'models/cvpr17-ILT/ILT-4_iter_40000.caffemodel'];
model_def = [caffe_root 'examples/cvpr17/deploy2.prototxt'];
phase = 'test';
net = caffe.Net(model_def, model_weights, phase);
% imgs = dir([img_root '*.JPEG']);
imgs = dir([img_root '*.jpg']);
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
    weight = out{2};
    out = out{1};
    
%     weight = permute(weight, [2,1,3]);
%     masked_conv6 = net.blobs('masked_conv6').get_data();
%     masked_conv6 = permute(masked_conv6, [2,1,3]);
% 
%     figure(101); imshow(resized_im);
%     weight = imresize(weight, [256, 256]);
%     masked_conv6 = imresize(masked_conv6, [256, 256]);
%     for c = 1: size(weight, 3)
%         figure(100);
%         subplot(2, 2, 1); imagesc(weight(:, :, c)); title('weight');
%         subplot(2, 2, 2); imagesc(masked_conv6(:, :, c));title('masked conv6');
%         subplot(2, 2, 3); imagesc(mat2gray(bsxfun(@times, single(resized_im), weight(:, :, c))));
%         subplot(2, 2, 4); imagesc(mat2gray(bsxfun(@times, single(resized_im), masked_conv6(:, :, c))));
%         pause;
%     end

%     

%     im_name = imgs(i).name(1:end-5);
%     ann = VOCreadxml([anno_root im_name '.xml']);
%     try
%         obj = zeros(length(ann.annotation.object), 1);
%     catch
%         continue;
%     end
%     for bb = 1 :length(ann.annotation.object)
%         obj(bb) = wnid2detid(ann.annotation.object(bb).name);
%     end
%     obj = unique(obj);
%    obj
%    
    gen_map = net.blobs('gen_map_sigmoid').get_data();
    gen_map = permute(gen_map, [2,1,3]);
    gen_map = imresize(gen_map, [256,256]);
%     figure(13);imshow((imresize(gen_map,[256,256])));%imagesc(gen_map);
  figure(15);subplot(1,2,1);imagesc(gen_map);
  figure(15);subplot(1,2,2);imagesc(mat2gray(bsxfun(@times, double(resized_im), double(gen_map>0.6))));
    feature_map = net.blobs('masked_conv6').get_data();
    feature_map = permute(feature_map, [2,1,3]);
    w = net.params('fc7',1).get_data();
    feature_map1 = reshape(feature_map, [], 512);
    cam = feature_map1 * w;
    cam = reshape(cam, 14, 14, 200);
%     cam = permute(cam, [2,1,3]);jkkkk
    figure(2);
    subplot(2,2,1);
    imagesc(im);
    subplot(2,2,2);
    [~, id] = max(out);
    act_id = find(out > 0.5);
    if isempty(act_id);
        act_id = id;
    end
%     imagesc(cam(:,:,id));
    imagesc(sum(cam(:,:,act_id),3));
    title(sprintf('%d: %s', id, synsets(id).name))
    figure(2); subplot(2,2,3);
    plot(out,'-o');
    ylim([0,1]);
    figure(2); subplot(2,2,4);
    imagesc(sum(feature_map, 3)); 
    a = max(max(feature_map));
    figure(12); plot(a(:), '-ro')
    conv6 = net.blobs('conv6').get_data;
 end
% imwrite(im, 'im.ppm');
caffe.reset_all();