function [superpixels, sp_num, affinity, feature] = OverSegment(im, model, opts)
assert(opts.num_scale == length(opts.k))
superpixels = cell(opts.num_scale, 1);
affinity = cell(opts.num_scale, 1);
feature = cell(opts.num_scale, 1);
sp_num = zeros(opts.num_scale, 1);
[E,~,~,segs]=edgesDetect(im,model);

for scale_id = 1:opts.num_scale
    opt = opts;
    opt.k = opt.k{scale_id};
    [cur_superpixels, V_rgb, V_lab] = spDetect(im,E,opt);
    cur_superpixels = 1 + cur_superpixels;
    [affinity{scale_id},~,~]=spAffinities(cur_superpixels,E,segs,opt.nThreads);
    cur_superpixels = double(cur_superpixels);
    sp_num(scale_id)=max(cur_superpixels(:));
    assert(sp_num(scale_id) == length(unique(cur_superpixels(:))));
    superpixels{scale_id} = cur_superpixels;
    V = cat(3, V_rgb, V_lab);
    feature{scale_id} = reshape(V, [], size(V, 3));
end

end