classdef GMM < handle
    %UNTITLED3 Summary of this class goes here
    %   Detailed explanation goes here
    
    properties(SetAccess=protected)
        num_cluster_ = 10;
        max_iter_ = 2;
        label_;
        feature_;
        N_;
        pi_;
        mu_;
        sig_;
        gaussian_pdf_;
        gamma_;
        dim_;
    end
    
    methods
        function obj = GMM(varargin)
            assert(nargin >= 1, 'Usage: a = GMM(features, [num_cluster], [max_iter])'); 
            obj.feature_ = varargin{1};
            obj.N_ = size(obj.feature_, 2);
            obj.dim_ = size(obj.feature_, 1);
            if nargin >= 2
            obj.num_cluster_ = varargin{2};
            end
            if nargin >= 3
                obj.max_iter_ = varargin{3};
            end
            obj.Init();
            for it = 1:obj.max_iter_
                obj.NextIteration();
            end
        end
        
        function Init(obj) 
            [obj.label_, obj.mu_] = kmeans(obj.feature_', obj.num_cluster_);
            obj.mu_ = obj.mu_';
            obj.pi_ = zeros(obj.num_cluster_, 1);
            for k = 1 : obj.num_cluster_
                feature_zero_mean = bsxfun(@minus, obj.feature_(:, obj.label_ == k), obj.mu_(:,k));
                N_k = sum(double(obj.label_ == k));
                obj.sig_{k} = 1 / N_k * (feature_zero_mean * feature_zero_mean');
                obj.pi_(k) = N_k / obj.N_;
            end
        end
        
        function NextIteration(obj)
                obj.ComputePDF();
                obj.MStep();
        end
        
        function ComputePDF(obj)
            obj.gaussian_pdf_ = zeros(obj.num_cluster_, obj.N_);
            for k = 1 : obj.num_cluster_
                % E-Step
                feature_zero_mean = bsxfun(@minus, obj.feature_, obj.mu_(:,k));
                gaussian_pdf_tmp = -0.5 * feature_zero_mean' / (obj.sig_{k} + 0.01 * eye(obj.dim_));
                obj.gaussian_pdf_(k, :) =  1 / ((2 * pi)^(obj.dim_/2) * det((obj.sig_{k} + 0.01 * eye(obj.dim_)))^0.5) * ...
                    exp(sum(gaussian_pdf_tmp' .* feature_zero_mean, 1));
            end
        end
            
        function MStep(obj)
                % M-Step
                obj.gamma_ = bsxfun(@times, obj.gaussian_pdf_, obj.pi_);
                obj.gamma_ = bsxfun(@rdivide, obj.gamma_, sum(obj.gamma_, 1));     
                % N_k
                N_k = sum(obj.gamma_, 2);
                % mu
                obj.mu_ = obj.feature_ * obj.gamma_';
                obj.mu_ = bsxfun(@rdivide, obj.mu_, N_k');
                % pi
                obj.pi_ = N_k / obj.N_;
                % sigma
                for k = 1 : obj.num_cluster_
                    feature_zero_mean = bsxfun(@minus, obj.feature_, obj.mu_(:,k));
                    obj.sig_{k} =1 / N_k(k) * bsxfun(@times, feature_zero_mean, obj.gamma_(k, :)) * feature_zero_mean';
                end
        end
        function prob = ComputeProb(obj, feature)
            assert(obj.dim_ == size(feature, 1))
            obj.feature_ = feature;
            obj.N_ = size(feature, 2);
            obj.ComputePDF();
            prob =  obj.pi_' * obj.gaussian_pdf_;
        end
        
    end
    
end

