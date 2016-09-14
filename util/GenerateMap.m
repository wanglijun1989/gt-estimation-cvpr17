function mul_scale_res = GenerateMap(im, superpixels, crf, scale_weight, sp_num)
fgd_prob = crf.prob_(2, :);
[height, width, ~] = size(im);
num_scale = length(sp_num);
mul_scale_res = zeros(height, width,num_scale);
sp_id_start =  0;
for scale_id = 1:num_scale
    res = zeros(height, width);
    for sp_id = 1 : sp_num(scale_id)
        ms_sp_id = sp_id + sp_id_start;
        sp_loc = superpixels{scale_id} == sp_id;
        res(sp_loc) = fgd_prob(ms_sp_id);%>median(log_pro);
    end
    mul_scale_res(:,:,scale_id) = res;
    sp_id_start = sp_id_start + sp_num(scale_id);
end
mul_scale_res = reshape(mul_scale_res, [], num_scale) * scale_weight(:);
mul_scale_res = reshape(mul_scale_res, height, width);
mul_scale_res = (mul_scale_res - min(mul_scale_res(:))) / (max(mul_scale_res(:)) - min(mul_scale_res(:)));