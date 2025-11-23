function out = bvar1_minnesota(M, H, hyper)
% BVAR(1) with Minnesota prior for a 2D macro block (inflation, real activity).
% Returns posterior means (mu, Phi, Sigma) and Cholesky IRFs up to horizon H.
%
% INPUTS
%   M     : T x 2 data (columns: [inflation, real activity]), ideally z-scored
%   H     : IRF horizon in months (e.g., 48)
%   hyper : struct with Minnesota hyperparameters (all optional):
%           .lam1 (overall tightness, default 0.30)
%           .lam3 (cross-tightness, default 0.50)
%           .lam4 (constant tightness, default 0.10)
%           .d    (lag decay, default 1)
%           .priorRW (true -> own-lag mean 1; false -> 0; default true)
%
% OUTPUT (struct "out")
%   .muM    : 2x1 intercepts (posterior mean)
%   .PhiMM  : 2x2 VAR(1) matrix (posterior mean)
%   .SigmaM : 2x2 innovation covariance (from residuals at posterior mean)
%   .Bpost  : 3x2 coefficient matrix [const, lag1(m1), lag1(m2)]
%   .IRF    : 2x2x(H+1) Cholesky IRFs (shock to j, response of i), horizon 0..H

if nargin<2 || isempty(H), H = 48; end
if nargin<3, hyper = struct; end
defaults = struct('lam1',0.30,'lam3',0.50,'lam4',0.10,'d',1,'priorRW',true);
f = fieldnames(defaults);
for k=1:numel(f)
    if ~isfield(hyper,f{k}), hyper.(f{k}) = defaults.(f{k}); end
end

[T,k] = size(M);
assert(k==2,'M must be T x 2');

% --- Regression objects (VAR(1)): m_t = c + A*m_{t-1} + e_t
Mlag = M(1:end-1,:);                 % (T-1) x 2
Mnow = M(2:end,:);                   % (T-1) x 2
X    = [ones(T-1,1) Mlag];           % (T-1) x 3   [const, m1_{t-1}, m2_{t-1}]
q    = size(X,2);                    % 3

% --- Scale for Minnesota prior: sigma_j from univariate AR(1) per series
sigma = zeros(2,1);
for j=1:2
    y = Mnow(:,j);  Z = [ones(T-1,1) Mlag(:,j)];
    bh = Z\y;  eh = y - Z*bh;
    sigma(j) = max(std(eh),1e-6);
end

% --- Prior mean B0 and variance V0 (diagonal) equation by equation
B0 = zeros(q,2);
if hyper.priorRW
    % own-lag mean 1
    B0(1+1,1) = 1;    % eq1, lag of var1
    B0(1+2,2) = 1;    % eq2, lag of var2
end

V0 = zeros(q,q,2);
for i=1:2
    V = zeros(q,1);
    % constant variance:
    V(1) = (hyper.lam1^2)*(hyper.lam4^2)*(sigma(i)^2);
    % lag-1 variances:
    for j=1:2
        l = 1;
        v = (hyper.lam1^2) * (sigma(i)^2/sigma(j)^2) / (l^(2*hyper.d));
        if j~=i, v = v * (hyper.lam3^2); end
        V(1+j) = v;
    end
    V0(:,:,i) = diag(max(V,1e-12));
end

% --- Posterior mean (ridge closed form) per equation
Bpost = zeros(q,2);
XtX = X.'*X;
for i=1:2
    Vi_inv = inv(V0(:,:,i));
    lhs = XtX + Vi_inv;
    rhs = (X.'*Mnow(:,i)) + Vi_inv*B0(:,i);
    Bpost(:,i) = lhs \ rhs;
end

% --- Residuals, covariance, read out mu,Phi
E = Mnow - X*Bpost;                 % (T-1) x 2
df = max((T-1) - q, 1);
SigmaM = (E.'*E)/df;

muM   = Bpost(1,:).';               % 2x1
PhiMM = Bpost(2:end,:).';           % 2x2

% --- IRFs (Cholesky identification; variable 1 ordered first)
% Response of state to a 1-s.d. Cholesky shock in variable j: P*e_j at h=0,
% and PhiMM^h * (P*e_j) at horizon h.
[Pc, pflag] = chol(SigmaM,'lower');
if pflag>0
    Pc = chol(SigmaM + 1e-8*eye(2),'lower');   % tiny jitter if needed
end
IRF = zeros(2,2,H+1);
IRF(:,:,1) = Pc;                                % h = 0
Ph = eye(2);
for h=1:H
    Ph = Ph * PhiMM;
    IRF(:,:,h+1) = Ph * Pc;
end

% --- Pack output
out.muM    = muM;
out.PhiMM  = PhiMM;
out.SigmaM = SigmaM;
out.Bpost  = Bpost;
out.IRF    = IRF;
end
