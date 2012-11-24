function [mu, sigma] = EM(X, comp_num, alpha)

K = comp_num;
[N, M] = size(X);

G = zeros(N, K);
G_old = zeros(N, K);
mu = rand(K, M);
sigma = zeros(M, M, K);
for k = 1:K
    sigma(:, :, k) = eye(M, M);
end;
t = 1;

% init latent variable p
p = zeros(K, 1) + 1 / K;

do
G_old = G; % save
% E-step
for n = 1:N
    for k = 1:K
        G(n, k) = p(k) * mvnpdf(X(n, :), mu(k, :), sigma(:, :, k));
    end;
    G(n, :) = G(n, :) ./ sum(G(n, :));
end;
    
H = sum(G, 1);
p = H / N;

for k = 1:K
    mu(k, :) = sum(repmat(G(:, k), 1, M) .* X, 1) / H(k);
    X_m = X - repmat(mu(k, :), N, 1);
    sigma(:, :, k) = X_m' * diag(G(:, k)) * X_m / H(k);
end;

t = max(max(min(abs(G - G_old), t)));

until(t < alpha)

end;