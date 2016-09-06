close all;
clear;
clc;
addpath(genpath('./external'))
addpath('util/');
addpath('/home/lijun/Research/Code/mexopencv/');
devkit_path = '/home/lijun/Research/DataSet/ILSVRC2014/ILSVRC2014_devkit/ILSVRC2014_devkit/';
addpath([devkit_path 'evaluation/'],'data_prepare/');
load([devkit_path 'data/meta_det.mat']);
wnid2detid = Wnid2Detid(synsets);
caffe_root = '/home/lijun/Research/Code/caffe-blvc/';
addpath([caffe_root '/matlab/'])
caffe.reset_all;
gpu_id = 0;
caffe.set_mode_gpu();
caffe.set_device(gpu_id);
model_weights = [caffe_root 'models/cvpr17-ILT/ILT-3_iter_12000.caffemodel'];
model_def = [caffe_root 'examples/cvpr17/deploy2.prototxt'];
phase = 'test';
net = caffe.Net(model_def, model_weights, phase);
fore_thr = 0.5;
fore_area_thr = 0.5;
fea_theta = 1e-2;
position_theta = 1e-2;
smooth_theta = 1e-2;
%%------------------------set parameters---------------------%%
theta=10; % control the edge weight 
alpha=0.99;% control the balance of two items in manifold ranking cost function

model=load('models/forest/modelBsds'); 
model=model.model;
model.opts.nms=-1; model.opts.nThreads=4;
model.opts.multiscale=0; model.opts.sharpen=2;

%% set up opts for spDetect (see spDetect.m)
opts = spDetect;
opts.nThreads = 4;  % number of computation threads
opts.k = 512;       % controls scale of superpixels (big k -> big sp)
opts.alpha = .5;    % relative importance of regularity versus data terms
opts.beta = .9;     % relative importance of edge versus color terms
opts.merge = 0.;%0;     % set to small value to merge nearby superpixels at end
%%

imgRoot='/home/lijun/Research/DataSet/Saliency/ECSSD/ECSSD-Image/';% test image path
% imgRoot= './';
% data_path = '/home/lijun/Research/DataSet/ILSVRC2014/ILSVRC2014_DET/';
% imgRoot = [data_path 'image/ILSVRC2013_DET_val/'];
res_path='./crf_gmm_res/';% the output path of the saliency map
mkdir(res_path);
imnames=dir([imgRoot '*' 'jpg']);

for ii=1:length(imnames)
    disp(ii);
%     imname=[imgRoot imnames(ii).name];
%     [input_im,w]=removeframe(imname);% run a pre-processing to remove the image frame
%     [m,n,k] = size(input_im);
    
    im = imread(sprintf('%s%s', imgRoot, imnames(ii).name));
    [height, width, ch] = size(im);
    if ch ~= 3
        im = repmat(im, [1,1,3]);
    end
    input = prepare_img(im, false);
    out = net.forward({input});
    out = out{1};
    gen_map = net.blobs('gen_map_sigmoid').get_data();
    gen_map = permute(gen_map, [2,1,3]);
    gen_map = imresize(gen_map, [height, width]);
    gen_map = double(gen_map > fore_thr);
    
    [E,~,~,segs]=edgesDetect(im,model);
    [superpixels, V_rgb, V_lab] = spDetect(im,E,opts);
    superpixels = 1 + superpixels;
    [affinity,~,~]=spAffinities(superpixels,E,segs,opts.nThreads);
    superpixels = double(superpixels);
    sp_num=max(superpixels(:));
    assert(sp_num == length(unique(superpixels(:))))
    fea_sp = nan(6, sp_num);
    position = nan(2, sp_num);
    init_label = zeros(1, sp_num);
    V = cat(3, V_rgb, V_lab);
    V = reshape(V, [], 6);
    %% label each superpixel
    for i = 1 : sp_num
        sp_loc = find(superpixels == i);
        fea_sp(:, i) = V(sp_loc(1), :);
        [r, c] = ind2sub([height, width], sp_loc);
        position(1, i) = mean(r/height);
        position(2, i) = mean(c/width);
        area = length(sp_loc);
        fore_area = sum(sum(gen_map(sp_loc)));
        init_label(i) = double(fore_area / area > fore_area_thr);
    end
    %% compute edge weights and construct CRF
    
    edge_feature = ComputeSimilarity(fea_sp, fea_theta);
    edge_position = ComputeSimilarity(position, position_theta);
    edge_smooth = ComputeSimilarity(position, smooth_theta);
    edge_appearance = edge_feature .* edge_position;
    affinity(1:size(affinity, 1)+1:end) = 0;
    edge_appearance(1:size(affinity, 1)+1:end) = 0;
    edge_smooth(1:size(affinity, 1)+1:end) = 0;
    boundary = superpixels(1, :)';
    boundary = [boundary; superpixels(end, :)'];
    boundary = [boundary; superpixels(:, 1)];
    boundary = [boundary; superpixels(:, end)];
    boundary = unique(boundary);
    edge_appearance = bsxfun(@rdivide, edge_appearance, sum(edge_appearance, 1));
    edge_smooth = bsxfun(@rdivide, edge_smooth, sum(edge_smooth, 1));
    
    crf = CRF([fea_sp; position], init_label, {affinity, edge_appearance, edge_smooth}, [0.1, 2, 1], boundary);
    %% Show GMM labeling
%     fgd_prob = crf.prob_(2, :);
%     res = zeros(height, width);
%     for i = 1 : sp_num
%         sp_loc = find(superpixels == i);
%         res(sp_loc) = fgd_prob(i);%>median(log_pro);
%     end
%     figure(1)
%     subplot(2,2,1); imshow(im);
%     subplot(2,2,2); imshow(gen_map);
%     subplot(2,2,3); imshow(mat2gray(res));
%     title('GMM results');
%     pause();
    %% CRF iteration
    for iteration = 1:5
        crf.NextIter();
    end
    fgd_prob = crf.prob_(2, :);
    res = zeros(height, width);
    for i = 1 : sp_num
        sp_loc = find(superpixels == i);
        res(sp_loc) = fgd_prob(i);%>median(log_pro);
    end
    
    res = (res - min(res(:))) / (max(res(:)) - min(res(:)));
    imwrite(res, [res_path imnames(ii).name(1:end-3) 'png']);
%     figure(1)
%     subplot(2,2,1); imshow(im);
%     subplot(2,2,2); imshow(gen_map);
%     subplot(2,2,3); imshow(mat2gray(res));
%     title(sprintf('Iteration %d/%d', iteration, 10));
%     pause();
    continue;
    %% 
    %% MR
    
    
    %%----------------------generate superpixels--------------------%%
    I = uint8(input_im * 255);
    [E,~,~,segs]=edgesDetect(I,model);
    [superpixels, V_rgb, V_lab] = spDetect(I,E,opts);
    superpixels = 1 + double(superpixels);
    if length(unique(superpixels(:))) ~= max(superpixels(:))
        1;
    end   
    
    %     superpixels=ReadDAT([m,n],spname); % superpixel label matrix
    spnum=max(superpixels(:));% the actual superpixel number
 

%%----------------------design the graph model--------------------------%%
% compute the feature (mean color in lab color space) 
% for each node (superpixels)
    input_vals=reshape(input_im, m*n, k);
    rgb_vals=zeros(spnum,1,3);
    inds=cell(spnum,1);
    for i=1:spnum
        inds{i}=find(superpixels==i);
        rgb_vals(i,1,:)=mean(input_vals(inds{i},:),1);
    end  
    lab_vals = colorspace('Lab<-', rgb_vals); 
    seg_vals=reshape(lab_vals,spnum,3);% feature for each superpixel
 
 % get edges
    adjloop=AdjcProcloop(superpixels,spnum);
    edges=[];
    for i=1:spnum
        indext=[];
        ind=find(adjloop(i,:)==1);
        for j=1:length(ind)
            indj=find(adjloop(ind(j),:)==1);
            indext=[indext,indj];
        end
        indext=[indext,ind];
        indext=indext((indext>i));
        indext=unique(indext);
        if(~isempty(indext))
            ed=ones(length(indext),2);
            ed(:,2)=double(i)*ed(:,2);
            ed(:,1)=indext;
            edges=[edges;ed];
        end
    end

% compute affinity matrix
    weights = makeweights(edges,seg_vals,theta);
    W = adjacency(edges,weights,spnum);

% learn the optimal affinity matrix (eq. 3 in paper)
    dd = sum(W); D = sparse(1:spnum,1:spnum,dd); clear dd;
    optAff =(D-alpha*W)\eye(spnum); 
    mz=diag(ones(spnum,1));
    mz=~mz;
    optAff=optAff.*mz;
  
%%-----------------------------stage 1--------------------------%%
% compute the saliency value for each superpixel 
% with the top boundary as the query
    Yt=zeros(spnum,1);
    bst=unique(superpixels(1,1:n));
    Yt(bst)=1;
    bsalt=optAff*Yt;
    bsalt=(bsalt-min(bsalt(:)))/(max(bsalt(:))-min(bsalt(:)));
    bsalt=1-bsalt;

% down
    Yd=zeros(spnum,1);
    bsd=unique(superpixels(m,1:n));
    Yd(bsd)=1;
    bsald=optAff*Yd;
    bsald=(bsald-min(bsald(:)))/(max(bsald(:))-min(bsald(:)));
    bsald=1-bsald;
   
% right
    Yr=zeros(spnum,1);
    bsr=unique(superpixels(1:m,1));
    Yr(bsr)=1;
    bsalr=optAff*Yr;
    bsalr=(bsalr-min(bsalr(:)))/(max(bsalr(:))-min(bsalr(:)));
    bsalr=1-bsalr;
  
% left
    Yl=zeros(spnum,1);
    bsl=unique(superpixels(1:m,n));
    Yl(bsl)=1;
    bsall=optAff*Yl;
    bsall=(bsall-min(bsall(:)))/(max(bsall(:))-min(bsall(:)));
    bsall=1-bsall;   
   
% combine 
    bsalc=(bsalt.*bsald.*bsall.*bsalr);
    bsalc=(bsalc-min(bsalc(:)))/(max(bsalc(:))-min(bsalc(:)));
    
% assign the saliency value to each pixel     
     tmapstage1=zeros(m,n);
     for i=1:spnum
        tmapstage1(inds{i})=bsalc(i);
     end
     tmapstage1=(tmapstage1-min(tmapstage1(:)))/(max(tmapstage1(:))-min(tmapstage1(:)));
     
     mapstage1=zeros(w(1),w(2));
     mapstage1(w(3):w(4),w(5):w(6))=tmapstage1;
     mapstage1=uint8(mapstage1*255);  

     outname=[saldir imnames(ii).name(1:end-4) '_stage1' '.png'];
     
%      imwrite(mapstage1,outname);

%%----------------------stage2-------------------------%%
% binary with an adaptive threhold (i.e. mean of the saliency map)
    th=mean(bsalc);
    bsalc(bsalc<th)=0;
    bsalc(bsalc>=th)=1;
    
% compute the saliency value for each superpixel
    fsal=optAff*bsalc;    
    
% assign the saliency value to each pixel
    tmapstage2=zeros(m,n);
    for i=1:spnum
        tmapstage2(inds{i})=fsal(i);    
    end
    tmapstage2=(tmapstage2-min(tmapstage2(:)))/(max(tmapstage2(:))-min(tmapstage2(:)));

    mapstage2=zeros(w(1),w(2));
    mapstage2(w(3):w(4),w(5):w(6))=tmapstage2;
    mapstage2=uint8(mapstage2*255);
    outname=[saldir imnames(ii).name(1:end-4) '_stage2' '.png'];   
%     imwrite(mapstage2,outname);
    figure(2);
    subplot(2,2,1); imshow(im); 
    subplot(2,2,2); imshow(tmapstage2); 
    subplot(2,2,3); imshow(gen_map); 
     subplot(2,2,4); imshow(mat2gray(bsxfun(@times, double(im), double(gen_map)))); 
end



