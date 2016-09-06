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
        boundary_
    end
    
    methods
        function obj = CRF(varargin)
            assert(nargin >= 3, 'Usage: a = CRF(feature, label, edge, [edge_weights])'); 
            assert(iscell(varargin{3}));
            obj.feature_ = varargin{1};
            obj.label_ = varargin{2};
            [obj.dim_, obj.N_] = size(obj.feature_);
            assert(obj.N_ == length(obj.label_));
            assert(max(obj.label_(:)) <= 1 && min(obj.label_(:) >=0), 'This is a binary CRF.');
            obj.edge_ = varargin{3};
            for i = 1 :length(obj.edge_)
                assert(sum(abs(diag(obj.edge_{i}))) == 0, 'nodes are not self-connected.');
            end
            if nargin >= 4
                obj.edge_weight_ = varargin{4};
                assert(length(obj.edge_) == length(obj.edge_weight_));
            else
                obj.edge_weight_ = 0.3 * ones(length(obj.edge_), 1);
            end
            if nargin >= 5
                obj.boundary_ = varargin{5};
                obj.label_(obj.boundary_) = 0;
            else
                obj.boundary_ = [];
            end
            obj.Init();
        end
        
        function Init(obj)
            obj.fgd_model_ = GMM(obj.feature_(:, obj.label_ > 0));
            obj.bgd_model_ = GMM(obj.feature_(:, obj.label_ <= 0));
            obj.prob_(1, :) = obj.bgd_model_.ComputeProb(obj.feature_);
            obj.prob_(2, :) = obj.fgd_model_.ComputeProb(obj.feature_);
            obj.prob_(2, obj.boundary_) = 0;
            obj.prob_(2, obj.label_ <= 0) = 0;
            obj.unary_ = -log(obj.prob_+0.1);
            obj.Z_ = sum(obj.prob_, 1);
            obj.prob_ = bsxfun(@rdivide, obj.prob_, obj.Z_);
        end
        
        function UpdateUnary(obj)
           
           obj.label_ = obj.prob_(2, :) > 0.5;
           obj.fgd_model_ = GMM(obj.feature_(1:6, obj.label_ > 0));
           obj.bgd_model_ = GMM(obj.feature_(1:6, obj.label_ <= 0));
           obj.unary_(1, :) = obj.bgd_model_.ComputeProb(obj.feature_(1:6,:));
           obj.unary_(2, :) = obj.fgd_model_.ComputeProb(obj.feature_(1:6,:));
           obj.unary_ = -log(obj.unary_);
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
            
            for i =  1 : 5
                obj.UpdateProb();
            end
            obj.UpdateUnary();
        end
    end
    
end

