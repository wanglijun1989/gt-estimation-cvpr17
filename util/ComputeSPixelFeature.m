function sp_info = ComputeSPixelFeature(superpixels, sp_num, label_map, feature, background_cue, im_size, fore_area_thr)
assert(iscell(superpixels) &&  iscell(feature))
num_scale = length(superpixels);
height = im_size(1); width = im_size(2);
sp_info.sp_num = sp_num;
sp_info.fea = cell(num_scale, 1);
sp_info.position = cell(num_scale, 1);
sp_info.init_label = cell(num_scale, 1);
sp_info.background_cue = cell(num_scale, 1);
for scale_id = 1:num_scale
    sp_info.fea{scale_id} = nan(6, sp_num(scale_id));
    sp_info.position{scale_id} = nan(2, sp_num(scale_id));
    sp_info.init_label{scale_id} = zeros(1, sp_num(scale_id));
    sp_info.background_cue{scale_id}  = zeros(2, sp_num(scale_id));
    for sp_id = 1 : sp_num(scale_id)
        sp_loc = find(superpixels{scale_id} == sp_id);
        sp_info.fea{scale_id}(:, sp_id) = feature{scale_id}(sp_loc(1), :);
        [r, c] = ind2sub([height, width], sp_loc);
        sp_info.position{scale_id}(1, sp_id) = mean(r/height);
        sp_info.position{scale_id}(2, sp_id) = mean(c/width);
        area = length(sp_loc);
        fore_area = sum(sum(label_map(sp_loc)));
        sp_info.background_cue{scale_id}(2, sp_id) = max(background_cue(sp_loc));
        sp_info.init_label{scale_id}(sp_id) = double(fore_area / area > fore_area_thr);
    end
    sp_info.background_cue{scale_id}(1,:) = 1 - sp_info.background_cue{scale_id}(2, :);
end
end