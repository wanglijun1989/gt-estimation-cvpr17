function res = visualization(im, gen_map, superpixels, crf, visualize, return_value)
res = [];
if visualize || return_value
    fgd_prob = crf.prob_(2, :);
    [height, width, ~] = size(im);
    res = zeros(height, width);
    sp_num = max(superpixels(:));
    for i = 1 : sp_num
        sp_loc = superpixels == i;
        res(sp_loc) = fgd_prob(i);%>median(log_pro);
    end
    res = (res - min(res(:))) / (max(res(:)) - min(res(:)));
end
if visualize 
    figure(1)
    subplot(2,2,1); imshow(im);
    subplot(2,2,2); imshow(gen_map);
    subplot(2,2,3); imshow(mat2gray(res));
    title('GMM results');
    pause
end