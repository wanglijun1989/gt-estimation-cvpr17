function [edge_appearance_whole, edge_smooth_whole, affinity_whole] = ComputeEdgeWeight(sp_info, affinity, opt)
num_scale = length(sp_info.fea);

edge_appearance = cell(num_scale, num_scale);
edge_smooth = cell(num_scale, num_scale);
for scale_y = 1:num_scale
    %% Compute edge weights within each scale
    edge_feature = ComputeSimilarity(sp_info.fea{scale_y}, opt.fea_theta(scale_y));
    edge_position = ComputeSimilarity(sp_info.position{scale_y}, opt.position_theta(scale_y));
    edge_appearance{scale_y, scale_y} = edge_feature .* edge_position;
    edge_smooth{scale_y, scale_y} = ComputeSimilarity(sp_info.position{scale_y}, opt.smooth_theta(scale_y));
    scale_x = scale_y+1;
    if scale_x > num_scale
        continue
    end
    %% Compute edge weights across consecutifve scales
    
    edge_feature = ComputeSimilarity(sp_info.fea{scale_y}, sp_info.fea{scale_x}, opt.fea_theta2);
    edge_position = ComputeSimilarity(sp_info.position{scale_y}, sp_info.position{scale_x}, opt.position_theta2);
    edge_smooth{scale_y, scale_x} = ComputeSimilarity(sp_info.position{scale_y}, sp_info.position{scale_x}, opt.smooth_theta2);
    edge_appearance{scale_y, scale_x} = edge_feature .* edge_position;
end
 %% Aggregation 
 edge_appearance_whole = edge_appearance{1};
 edge_smooth_whole = edge_smooth{1};
 affinity_whole = affinity{1};
 [height_last, width_last] = size(edge_smooth_whole);
 for scale_id = 2 : num_scale
     % edge weights within each scale
     [height_cur, width_cur] = size(edge_appearance{scale_id, scale_id});
     edge_appearance_whole(end+1:end+ height_cur, end+1:end+width_cur) =...
         edge_appearance{scale_id, scale_id};
     edge_smooth_whole(end+1:end+ height_cur, end+1:end+width_cur) =...
         edge_smooth{scale_id, scale_id}; 
     affinity_whole(end+1:end+ height_cur, end+1:end+width_cur) =...
         affinity{scale_id};
     % edge weights across consecutifve scales
     edge_smooth_whole(end-height_last+1:end, end-width_cur + 1 : end) = edge_smooth{scale_id -1, scale_id};
     edge_smooth_whole(end-height_cur+1:end, end-width_last+1:end) = edge_smooth{scale_id -1, scale_id}';
     edge_smooth_whole(end-height_last+1:end, end-width_cur + 1 : end) = edge_smooth{scale_id -1, scale_id};
     edge_smooth_whole(end-height_cur+1:end, end-width_last+1:end) = edge_smooth{scale_id -1, scale_id}';
     height_last = height_cur; width_last = width_cur;
 end
       

end

