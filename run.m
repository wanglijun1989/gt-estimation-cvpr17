% it = [3000, 3500, 4000, 4500, 5000, 5500, 6000, 6500, 7000, 7500, 8000, 10000, 20000, 30000, 50000];
it = [5000, 8000, 10000, 15000, 20000, 30000, 50000, 60000];
thr = [0.65, 0.68, 0.7, 0.72];
post = {'65', '68', '70', '72', '74', '76'};
for iter = 1:length(it)
for t = 1:length(thr)
iter_num=num2str(it(iter));
postfix = post{t};
crf_opt.fore_thr = thr(t);
try
test_gc
catch
fprintf('failed it: %s, thr: %s\n', iter_num, postfix);
end
end
end