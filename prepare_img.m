function out = prepare_img(im)
im_mean = load('model/ilsvrc_2012_mean.mat');
IMAGE_DIM = 256;
im_data = im(:, :, [3, 2, 1]);  % permute channels from RGB to BGR
im_data = permute(im_data, [2, 1, 3]);  % flip width and height
im_data = single(im_data);  % convert from uint8 to single
im_data = imresize(im_data, [IMAGE_DIM IMAGE_DIM], 'bilinear');  % resize im_data
out = im_data - im_mean.mean_data;  % subtract mean_data (already in W x H x C, BGR)
out = imresize(out, [224 224], 'bilinear');  % resize im_data
end