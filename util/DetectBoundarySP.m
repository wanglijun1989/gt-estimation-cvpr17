function boundary = DetectBoundarySP(superpixels)
boundary = superpixels(1, :)';
boundary = [boundary; superpixels(end, :)'];
boundary = [boundary; superpixels(:, 1)];
boundary = [boundary; superpixels(:, end)];
boundary = unique(boundary);
end