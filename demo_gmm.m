close all; clear; clc;
feature(:, 1:100) = randn(3,100);
feature(:, 101:200) = randn(3,100) + 1;
feature(:, 201:300) = randn(3,100) + 3;
feature(:, 301:400) = randn(3,100) + 5;

gmm = GMM(feature, 4, 1);

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