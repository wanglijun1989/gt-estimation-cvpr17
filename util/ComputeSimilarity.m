function sim = ComputeSimilarity(varargin)
assert(nargin >= 1, 'Usage: sim = ComputeSimilarity(feature, [theta]), where each column of feature is a sample');

feature = varargin{1};
N = size(feature, 2);

if nargin > 1
    theta = varargin{2};
else
    theta = 1;
end

%% compute euclidean distance
x2 = sum(feature.^2, 1);
x2 = repmat(x2, N, 1);
y2 = x2';

xy = feature' * feature;

eudist = x2 + y2 - 2 * xy;
sim = exp(-eudist/2/theta);
end