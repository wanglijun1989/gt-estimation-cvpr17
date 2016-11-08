devkit_path = '/home/lijun/Research/DataSet/ILSVRC2014/ILSVRC2014_devkit/ILSVRC2014_devkit/';
data_path = '/home/lijun/Research/DataSet/ILSVRC2014/ILSVRC2014_DET/';
img_root = [data_path 'image/ILSVRC2014_DET_train/'];
map_root = [data_path 'sal_map_2_png_merged/sal_map_2_png/ILSVRC2014_DET_train/'];
anno_root = [data_path 'BOX/ILSVRC2014_DET_bbox_train/'];
addpath([devkit_path 'evaluation/']);
load([devkit_path 'data/meta_det.mat']);
wnid2detid = Wnid2Detid(synsets);

sub_dir = dir(map_root);
fid = fopen('train_sal_img.txt', 'w+');
obj_hist = zeros(20, 1);
for sub_id = 1:length(sub_dir)
    if ~isdir([img_root sub_dir(sub_id).name]) || strcmp(sub_dir(sub_id).name, '.') || strcmp(sub_dir(sub_id).name, '..')
        continue;
    end
    fprintf('Processing synset: %d/%d\n', sub_id, length(sub_dir));
    img_path = [img_root sub_dir(sub_id).name '/'];
    anno_path = [anno_root sub_dir(sub_id).name '/'];
    map_path = [map_root sub_dir(sub_id).name '/'];
    maps = dir([map_path '*.png']);
    for img_id = 1:length(maps)
        im_name = maps(img_id).name(1:end-4);
        ann = VOCreadxml([anno_path im_name '.xml']);
        if isfield(ann.annotation, 'object')
            obj = zeros(length(ann.annotation.object), 1);
            for bb = 1 :length(ann.annotation.object)
                obj(bb) = wnid2detid(ann.annotation.object(bb).name);
            end
            obj = unique(obj);
            obj_hist(length(obj)) = obj_hist(length(obj)) + 1;
            if length(obj) > 3
                continue;
            end
        end
        
        fprintf(fid, '%s/%s 0\n', sub_dir(sub_id).name, maps(img_id).name);
    end
end
fclose(fid);
%% shuffling data
res = importdata('train_sal_img.txt');
num_imgs = length(res.textdata);
rand_index = randperm(num_imgs);
img_fid = fopen('train_sal_2_png_img_shuffled.txt', 'w+');
map_fid = fopen('train_sal_2_png_map_shuffled.txt', 'w+');
for img_id = 1:num_imgs
    fprintf(map_fid, '%s 0\n', res.textdata{rand_index(img_id)});
    fprintf(img_fid, '%s 0\n', [res.textdata{rand_index(img_id)}(1:end-3) 'JPEG']);
end
fclose(map_fid);
fclose(img_fid);



