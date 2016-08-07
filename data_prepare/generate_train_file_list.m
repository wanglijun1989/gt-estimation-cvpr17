devkit_path = '/home/lijun/Research/DataSet/ILSVRC2014/ILSVRC2014_devkit/ILSVRC2014_devkit/';
data_path = '/home/lijun/Research/DataSet/ILSVRC2014/ILSVRC2014_DET/';
img_root = [data_path 'image/ILSVRC2014_DET_train/'];
anno_root = [data_path 'BOX/ILSVRC2014_DET_bbox_train/'];
addpath([devkit_path 'evaluation/']);
load([devkit_path 'data/meta_det.mat']);
wnid2detid = Wnid2Detid(synsets);

sub_dir = dir(img_root);
fid = fopen('train.txt', 'w+');
for sub_id = 1:length(sub_dir)
    if ~isdir([img_root sub_dir(sub_id).name]) || strcmp(sub_dir(sub_id).name, '.') || strcmp(sub_dir(sub_id).name, '..')
        continue;
    end
    fprintf('Processing synset: %d/%d\n', sub_id, length(sub_dir));
    img_path = [img_root sub_dir(sub_id).name '/'];
    anno_path = [anno_root sub_dir(sub_id).name '/'];
    imgs = dir([img_path '*.JPEG']);
    for img_id = 1:length(imgs)
        im_name = imgs(img_id).name(1:end-5);
        ann = VOCreadxml([anno_path im_name '.xml']);
        if isfield(ann.annotation, 'object')
            obj = zeros(length(ann.annotation.object), 1);
            for bb = 1 :length(ann.annotation.object)
                obj(bb) = wnid2detid(ann.annotation.object(bb).name);
            end
            obj = unique(obj);
            fprintf(fid, '%s%s%s\n', sub_dir(sub_id).name, '/', imgs(img_id).name);
            for bb = 1:length(obj)
                fprintf(fid, '%d ', obj(bb));
            end
            fprintf(fid, '%d\n', 0);
        end
    end
end

fclose(fid);
