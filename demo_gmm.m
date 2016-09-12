close all; clear; clc;
feature(:, 1:1000) = randn(3,1000);
feature(:, 1001:2000) = randn(3,1000) + 1;
feature(:, 2001:3000) = 2*randn(3,1000) + 3;
feature(:, 3001:4000) = 3*randn(3,1000) + 5;

gmm = GMM(200* feature, 4, 1);

color = rand(gmm.num_cluster_, 3);
ecolor = rand(gmm.num_cluster_,3);

[~, id] = max(gmm.gamma_, [], 1);

for i = 1:size(feature, 2)
figure(1);hold on;
plot3(feature(1, i), feature(2, i), feature(3, i), 'markerfacecolor', color(id(i), :), ...
    'markeredgecolor', ecolor(id(i), :),'markersize',25,'marker', 'o', ...
    'linewidth', 4);
end
grid on

for it = 1:10
    gmm.NextIteration();
    [~, id] = max(gmm.gamma_, [], 1);
    clf
    for i = 1:size(feature, 2)
        figure(1);hold on;
        plot3(feature(1, i), feature(2, i), feature(3, i), 'markerfacecolor', color(id(i), :), ...
            'markeredgecolor', ecolor(id(i), :),'markersize',25,'marker', 'o', ...
            'linewidth', 4);
    end
    grid on
end