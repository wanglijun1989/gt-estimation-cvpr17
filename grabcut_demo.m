clear
clc
addpath('/home/lijun/Research/Code/mexopencv/');
img_root = '/home/lijun/Research/DataSet/Saliency/ECSSD/ECSSD-Image/';
imgs = dir([img_root '*.jpg']);
num_imgs = length(imgs);
for i = 52:num_imgs
    im = imread(sprintf('%s%s', img_root, imgs(i).name));
    flag = true;
    fh = figure(12);
    imshow(im);
    while flag
       bb  = getrect(fh);
       map = cv.grabCut(im, bb, 'MaxIter', 1); 
       figure(13); 
       imshow(mat2gray(map));
       figure(12);
       [~, ~, id] = ginput(1);
       if id ~= 1
           flag = false;
       end
    end
    
end