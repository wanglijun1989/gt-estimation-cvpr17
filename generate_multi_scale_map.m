scale1_path = 'crf_gmm_res/';
scale2_path = 'crf_gmm_res_1024/';
scale3_path = 'crf_gmm_res_2048/';
multi_scale_path = 'crf_gmm_res_multi_scale/';
maps_file = dir([scale1_path '*png']);

for i = 1:length(maps_file)
    map1 = mat2gray(imread([scale1_path maps_file(i).name]));
    map2 = mat2gray(imread([scale2_path maps_file(i).name]));
    map3 = mat2gray(imread([scale3_path maps_file(i).name]));
    map = map1*0.4 + map2*0.3 + map3*0.3;
    map = (map-min(map(:))) / (max(map(:)) - min(map(:)));
    imwrite(map, [multi_scale_path maps_file(i).name]);
end