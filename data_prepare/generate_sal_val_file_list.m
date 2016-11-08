data_path = '/home/lijun/Research/DataSet/Saliency/PASCAL-S/';
img_root = [data_path 'PASCAL-S-Image/'];
map_root = [data_path 'PASCAL-S-Mask-1/'];
maps = dir([map_root '*png']);
img_fid = fopen('val_sal_img.txt', 'w+');
map_fid = fopen('val_sal_map.txt', 'w+');
for img_id = 1:length(maps)
    im_name = maps(img_id).name(1:end-5);
    fprintf(img_fid, '%s%s 0\n', 'PASCAL-S-Image/', [maps(img_id).name(1:end-3) 'jpg']);
    fprintf(map_fid, '%s%s 0\n', 'PASCAL-S-Mask-1/', maps(img_id).name);
end
fclose(img_fid);
fclose(map_fid);




