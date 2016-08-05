function [ w ] = project_simplex(v)
n = length(v);
U = 1:n;
n_U = n;
ro = 0;
s = 0;
while n_U > 0
    k = U(ceil(n_U/2));
    [~, G] = find(v(U) >= v(k));
    G = U(G);
    [~, L] = find(v(U) < v(k));
    L = U(L);
    dro = length(G);
    ds = v(G);
    ds = sum(ds);
    if s+ds - (ro+dro)*v(k) < 1
        s = s + ds; 
        ro = ro + dro;
        U = L;
    else
        U = G(G~=k);
    end
    n_U = length(U);
%     fprintf('n_U = %d \n', n_U);
end
theta = (s-1)/ro;
w = double(v-theta > 0).*(v-theta);

