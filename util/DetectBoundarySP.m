function boundary = DetectBoundarySP(superpixels, sp_num)
num_scale = length(sp_num);
boundary = [];
sp_start_id = 0;
for scale_id = 1: num_scale
    cur_seg = superpixels{scale_id};
    cur_boundary = [cur_seg(1, :)'+sp_start_id;
        cur_seg(end, :)'+sp_start_id;
        cur_seg(:, 1)+sp_start_id;
        cur_seg(:, end)+sp_start_id];
    cur_boundary = unique(cur_boundary);
    boundary = [boundary; cur_boundary];
    sp_start_id = sp_start_id + sp_num(scale_id);
end

end