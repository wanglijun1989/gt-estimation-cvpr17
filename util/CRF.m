classdef CRF < handle
    %UNTITLED Summary of this class goes here
    %   Detailed explanation goes here
    
    properties (SetAccess = protected)
        fgd_model_;
        bgd_model_;
        N_;
        dim_;
        feature_;
        label_;
        unary_;
        edge_;
        edge_weight_;
        prob_;
        Z_;
        boundary_;
        prior_;
        prior_weight_;
        sp_num_;
        group_num_
    end
    
    methods
        function obj = CRF(feature, label, edge, varargin)
            p = inputParser;
            p.addRequired('feature',@ismatrix);
            p.addRequired('label', @ismatrix);
            p.addRequired('edge', @iscell);
            p.addOptional('edge_weight', [], @ismatrix);
            p.addParameter('boundary',[], @ismatrix);
            p.addParameter('prior',[], @ismatrix);
            p.addParameter('prior_weight',0, @isnumeric);
            p.addParameter('sp_num',[], @ismatrix);
            
            p.parse(feature, label, edge, varargin{:});
            %% feaqture, label and edge
            obj.feature_ = p.Results.feature;
            obj.label_ = p.Results.label;
            obj.edge_ = p.Results.edge;
            [obj.dim_, obj.N_] = size(obj.feature_);
            % validation  
            assert(obj.N_ == length(obj.label_));
            assert(max(obj.label_(:)) <= 1 && min(obj.label_(:) >=0), 'This is a binary CRF.');
            for i = 1 :length(obj.edge_)
                assert(sum(abs(diag(obj.edge_{i}))) == 0, 'nodes are not self-connected.');
            end
            %% edge_weight
            obj.edge_weight_ = p.Results.edge_weight;
            if isempty(obj.edge_weight_)
                obj.edge_weight_ = 0.3 * ones(length(obj.edge_), 1);
            end
            assert(length(obj.edge_) == length(obj.edge_weight_));
            %% boundary
            obj.boundary_ = p.Results.boundary;
            obj.label_(obj.boundary_) = 0;
            %% piror
            obj.prior_ = p.Results.prior;
            if isempty(obj.prior_)
                obj.prior_= zeros(2, obj.N_);
            end
            %% prior weights
            obj.prior_weight_ = p.Results.prior_weight;
            %% sp_num
            obj.sp_num_ = p.Results.sp_num;
            if isempty(obj.sp_num_)
                obj.sp_num_ = obj.N_;
            end
            obj.group_num_ = length(obj.sp_num_);
            obj.Init();
        end
        
        function Init(obj)
            obj.fgd_model_ = cell(obj.group_num_, 1);
            obj.bgd_model_ = cell(obj.group_num_, 1);
            end_id = 0;
            for i = 1:obj.group_num_
                start_id = end_id+1;
                end_id = end_id + obj.sp_num_(i);
                cur_feature = obj.feature_(:, start_id:end_id);
                cur_label = obj.label_(:, start_id:end_id);
                obj.fgd_model_{i} = GMM(cur_feature(:, cur_label> 0));
                obj.bgd_model_{i} = GMM(cur_feature(:, cur_label<= 0));
                obj.prob_(1, start_id:end_id) = obj.bgd_model_{i}.ComputeProb(cur_feature);
                obj.prob_(2, start_id:end_id) = obj.fgd_model_{i}.ComputeProb(cur_feature);
            end
            obj.prob_ = bsxfun(@rdivide, obj.prob_, sum(obj.prob_, 1));
            obj.prob_ = obj.prob_ + obj.prior_weight_ * obj.prior_;
            obj.prob_(2, obj.boundary_) = 0;
%             obj.prob_(2, obj.label_ <= 0) = 0;
            obj.unary_ = -log(obj.prob_+0.1 + obj.prior_weight_ * obj.prior_);
            obj.Z_ = sum(obj.prob_, 1);
            obj.prob_ = bsxfun(@rdivide, obj.prob_, obj.Z_);
%             obj.UpdateUnary();
            obj.UpdateProb();
        end
        
        function UpdateUnary(obj)
           
           obj.label_ = obj.prob_(2, :) > 0.5;
           
           end_id = 0;
           for i = 1:obj.group_num_
               start_id = end_id+1;
               end_id = end_id + obj.sp_num_(i);
               cur_feature = obj.feature_(1:6, start_id:end_id);
               cur_label = obj.label_(:, start_id:end_id);
               obj.fgd_model_{i} = GMM(cur_feature(:, cur_label> 0));
               obj.bgd_model_{i} = GMM(cur_feature(:, cur_label<= 0));
               obj.unary_(1, start_id:end_id) = obj.bgd_model_{i}.ComputeProb(cur_feature);
               obj.unary_(2, start_id:end_id) = obj.fgd_model_{i}.ComputeProb(cur_feature);
           end
           obj.unary_ = bsxfun(@rdivide, obj.unary_, sum(obj.unary_, 1));
           obj.unary_ = -log(obj.unary_ +0.1 + obj.prior_weight_ * obj.prior_);
        end
        
        function UpdateProb(obj)
            message_tmp = zeros(size(obj.unary_));
            for i = 1 : length(obj.edge_)
                message_tmp(1, :) = message_tmp(1, :) + obj.edge_weight_(i) * obj.prob_(2, :) * obj.edge_{i};
                message_tmp(2, :) = message_tmp(2, :) + obj.edge_weight_(i) * obj.prob_(1, :) * obj.edge_{i};
                % Note: we have ensured that the node are not self
                % connected, i.e., obj.edge_{i}(j,j) == 0
            end
            obj.prob_ = exp(- obj.unary_ - message_tmp);
            obj.prob_(2, obj.boundary_) = 0;
            obj.Z_ = sum(obj.prob_, 1);
            obj.prob_ = bsxfun(@rdivide, obj.prob_, obj.Z_);
        end
        function NextIter(obj)
              obj.UpdateUnary();
            for i =  1 : 5
                obj.UpdateProb();
            end
        end
    end
    
end

