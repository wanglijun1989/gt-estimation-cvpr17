function sim = ComputeSimilarity(varargin)
assert(nargin >= 2, ['Usage: ',...
    'sim = ComputeSimilarity(feature, [theta]), or',...
    'sim = ComputeSimilarity(featurea, featureb, [theta]), or',...
    'where each column of feature is a sample']);
if nargin == 2
    feature = varargin{1};
    N = size(feature, 2);
    theta = varargin{2};
    %% compute euclidean distance
    x2 = sum(feature.^2, 1);
    x2 = repmat(x2, N, 1);
    y2 = x2';
    xy = feature' * feature;
    eudist = x2 + y2 - 2 * xy;
    sim = exp(-eudist/2/theta);
    return;
elseif nargin == 3
    featureA = varargin{1};
    NA = size(featureA, 2);
    featureB = varargin{2};
    NB = size(featureB, 2);
    theta = varargin{3};
    x2 = sum(featureA.^2, 1);
    x2 = repmat(x2', 1, NB);
    y2 = sum(featureB.^2, 1);
    y2 = repmat(y2, NA, 1);
    xy = featureA' * featureB;
    eudist = x2 + y2 - 2* xy;
    sim = exp(-eudist/2/theta);
    return;
end
end