devkit_path = '/home/lijun/Research/DataSet/ILSVRC2014/ILSVRC2014_devkit/ILSVRC2014_devkit/';
data_path = '/home/lijun/Research/DataSet/ILSVRC2014/ILSVRC2014_DET/';
img_root = [data_path 'image/ILSVRC2013_DET_val/'];
anno_root = [data_path 'BOX/ILSVRC2013_DET_bbox_val/'];
addpath([devkit_path 'evaluation/']);
load([devkit_path 'data/meta_det.mat']);
wnid2detid = Wnid2Detid(synsets);
fid = fopen('val.txt', 'w+');
imgs = dir([img_root '*.JPEG']);
for img_id = 1:length(imgs)
    im_name = imgs(img_id).name(1:end-5);
    ann = VOCreadxml([anno_root im_name '.xml']);
    if isfield(ann.annotation, 'object')
        obj = zeros(length(ann.annotation.object), 1);
        for bb = 1 :length(ann.annotation.object)
            obj(bb) = wnid2detid(ann.annotation.object(bb).name);
        end
        obj = unique(obj);
        fprintf(fid, '%s\n', imgs(img_id).name);
        for bb = 1:length(obj)
            fprintf(fid, '%d ', obj(bb));
        end
        fprintf(fid, '%d\n', 0);
    end
end


fclose(fid);
