function out = ap_two_step_bvar_p8(Y_mon, M, nList, hyper, floor_bp_in)
% ap_two_step_bvar_p8  Two-step ATSM with Step-1 Minnesota BVAR(8) and Step-2 MLE (latent)
%
% INPUTS:
%   Y_mon      : T x J yields in monthly decimals (annual% / 100 / 12)
%   M          : T x 2 macro factors (z-scored): [inflation, real activity]
%   nList      : 1 x J maturities in months
%   hyper      : (optional) struct with Minnesota hyperparams + short-rate ridge:
%                .lam1=0.25, .lam3=0.5, .lam4=0.1, .d=1, .priorRW=true, .lam_sr=0
%   floor_bp_in: (optional) 1 x J floors on measurement std (bp/year)
%
% OUTPUT (key fields):
%   out.step1.fixed     : struct with mu_o, Phi_o (companion), SigmaM (K1xK1), delta0, delta11
%   out.step2.params    : struct with RHO, lambda0, lambda1_MM, lambda1_LL, delta12, logRy_diag
%   out.fit.rmse_bp     : 1 x J RMSE (annualized bp) computed on trimmed sample (T-p)
%   out.fit.meas_std_bp : 1 x J measurement std (annualized bp)
%   out.latent.smooth   : 3 x (T-p) smoothed latent factors
%   out.meta            : floors, nList, p

% --------------------------- Checks & defaults ---------------------------
[T,J] = size(Y_mon);
assert(size(M,1)==T && size(M,2)==2, 'M must be T x 2');
nList = nList(:);  J = numel(nList);

if nargin < 4 || isempty(hyper), hyper = struct(); end
def = struct('lam1',0.25,'lam3',0.5,'lam4',0.1,'d',1,'priorRW',true,'lam_sr',0);
fns = fieldnames(def);
for k=1:numel(fns), if ~isfield(hyper,fns{k}), hyper.(fns{k}) = def.(fns{k}); end, end

% --------------------------- STEP 1: Minnesota BVAR(p=8) -----------------
K1 = 2;                          % macro dims
p  = 8;                          % VAR order
[mu_o, Phi_o, SigmaM, Mnow] = bvar_minnesota_p(M, p, hyper);   % Ko=K1*p companion

% Short rate on macro only (aligned with Mnow, length T-p)
r  = Y_mon(p+1:end, 1);
Xr = [ones(T-p,1), Mnow];  % contemporaneous macro only
if hyper.lam_sr > 0
    L = hyper.lam_sr * eye(size(Xr,2)); L(1,1)=0;
    b = (Xr.'*Xr + L) \ (Xr.'*r);
else
    b = Xr \ r;
end
delta0  = b(1);
delta11 = b(2:1+K1);

fixed.mu_o    = mu_o;       % Ko x 1 (Ko=K1*p)
fixed.Phi_o   = Phi_o;      % Ko x Ko
fixed.SigmaM  = SigmaM;     % K1 x K1 (innovation cov of m_t)
fixed.delta0  = delta0;
fixed.delta11 = delta11;

% --------------------------- STEP 2: data (trim to T-p) ------------------
data.Y      = Y_mon(p+1:end, :);   % (T-p) x J
data.M_full = M;                   % keep full macro to form companion state on the fly
data.To     = T - p;
data.J      = J;
data.nList  = nList;
data.p      = p;
data.K1     = K1;
data.Ko     = K1 * p;

% Floors (bp/year) -> monthly variances
if nargin >= 5 && ~isempty(floor_bp_in)
    assert(numel(floor_bp_in)==J, 'floor_bp_in length must be J');
    floor_bp = floor_bp_in(:);
else
    floor_bp = auto_floors_bp(nList);
end
data.v_floor = (floor_bp*1e-4/12).^2;

% --------------------------- STEP 2: Optimize latent/pricing -------------
theta0 = init_guess_generic(J);                    % same 27 + J unknowns
theta0(end-J+1:end) = log(max(data.v_floor(:), 1e-10));

obj = @(th) negloglik_latent_p(th, data, fixed);
opts = optimoptions('fminunc', 'Display','iter', 'Algorithm','quasi-newton', ...
    'FiniteDifferenceStepSize',1e-4, 'MaxIterations',2000, ...
    'MaxFunctionEvaluations',5e5, 'OptimalityTolerance',1e-6, ...
    'StepTolerance',1e-8, 'FunctionTolerance',1e-8);

[theta_hat, nll, exitflag] = fminunc(obj, theta0, opts); %#ok<ASGLU>
params = unpack_generic(theta_hat, J);

% --------------------------- Smoother & fit ------------------------------
[S, Yhat, meas_std_bp] = smooth_and_fit_p(fixed, params, data);

err = data.Y - Yhat;                               % monthly decimals, trimmed sample
rmse_ann_bp  = sqrt(mean(err.^2, 1)) * 120000;     % annualized bp

% --------------------------- Output --------------------------------------
out.step1.fixed      = fixed;
out.step2.params     = params;
out.step2.nll        = nll;
out.step2.exitflag   = exitflag;
out.fit.Yhat         = Yhat;
out.fit.rmse_bp      = rmse_ann_bp;
out.fit.meas_std_bp  = meas_std_bp;
out.latent.smooth    = S;
out.meta.floors_bp   = floor_bp(:).';
out.meta.nList       = nList(:).';
out.meta.p           = p;
end

% ======================================================================
function [mu_o, Phi_o, SigmaM, Mnow] = bvar_minnesota_p(M, p, hyper)
% Minnesota BVAR(p) for K1=2 macro, companion output
[T,K1] = size(M);  assert(K1==2);
To = T - p;        assert(To >= 30, 'Not enough data for VAR(%d).', p);

% Targets m_t for t=p+1..T
Mnow = M(p+1:end, :);                 % To x 2

% Regressors X = [1, m_{t-1},...,m_{t-p}]
q = 1 + K1*p;
X = ones(To, q);
for l=1:p
    X(:, 1 + (K1*(l-1)+(1:K1))) = M((p+1-l):(T-l), :);
end

% Scale parameters sigma_j from univariate AR(1) (robust, simple)
sigma = zeros(K1,1);
Mlag1 = M(1:end-1,:); Mnow1 = M(2:end,:);
for j=1:K1
    y  = Mnow1(:,j);
    Z  = [ones(T-1,1) Mlag1(:,j)];
    bh = Z\y; eh = y - Z*bh;
    sigma(j) = max(std(eh), 1e-6);
end

% Prior mean B0 and variance V0 (diagonal) per equation i
B0 = zeros(q, K1);
if hyper.priorRW
    % RW prior: own-lag at l=1 has mean 1
    % Position in X: 1 + (K1*(1-1) + i) = 1 + i
    B0(1+1,1) = 1;   % eq1 on var1 lag1
    B0(1+2,2) = 1;   % eq2 on var2 lag1
end

V0 = zeros(q,q,K1);
for i=1:K1
    V = zeros(q,1);
    % Intercept variance
    V(1) = (hyper.lam1^2) * (hyper.lam4^2) * (sigma(i)^2);
    % Lag coefficients
    for l=1:p
        for j=1:K1
            idx = 1 + (K1*(l-1) + j);
            v = (hyper.lam1^2) * (sigma(i)^2 / sigma(j)^2) / (l^(2*hyper.d));
            if j ~= i, v = v * (hyper.lam3^2); end
            V(idx) = max(v, 1e-12);
        end
    end
    V0(:,:,i) = diag(V);
end

% Posterior mean (ridge-by-equation)
XtX = X.'*X;  Bpost = zeros(q, K1);
for i=1:K1
    Vi_inv = inv(V0(:,:,i));
    lhs = XtX + Vi_inv;
    rhs = X.'*Mnow(:,i) + Vi_inv*B0(:,i);
    [L,pd] = chol(lhs,'lower');
    if pd==0
        Bpost(:,i) = L'\(L\rhs);
    else
        Bpost(:,i) = lhs \ rhs;
    end
end

% Innovations covariance from posterior mean residuals
E  = Mnow - X*Bpost;
df = max(To - q, 1);
SigmaM = (E.'*E) / df;               % K1 x K1

% Companion form (Ko = K1*p)
Ko = K1*p;
Phi_o = zeros(Ko, Ko);
PhiBlocks = Bpost(2:end,:).';         % K1 x (K1*p) = [Phi1 ... Phip]
Phi_o(1:K1, :) = PhiBlocks;
Phi_o(K1+1:Ko, 1:Ko-K1) = eye(Ko-K1);
mu_o = zeros(Ko,1); mu_o(1:K1) = Bpost(1,:).';

% Gentle stabilization if needed (spectral radius < 1)
eigPhi = max(abs(eig(Phi_o)));
if eigPhi >= 0.999
    Phi_o = Phi_o * (0.995/eigPhi);
    % Keep mu_o as-is; stationarity corrected by scaling Phi
end
end

% ======================================================================
function nll = negloglik_latent_p(theta, data, fixed)
% Step-2 likelihood with macro companion state (Ko = K1*p) and 3 latent
P  = unpack_generic(theta, data.J);
K1 = data.K1;  Ko = data.Ko;  K2 = 3;  K = Ko + K2;
J  = data.J;   To = data.To;  p  = data.p;

% P-measure blocks
PhiP = blkdiag(fixed.Phi_o, P.RHO);     % K x K
muP  = [fixed.mu_o; zeros(K2,1)];       % K x 1

% Shock square-root: only top K1 rows innovate in macro block
Lm = chol(fixed.SigmaM,'lower');        % K1 x K1
Laug = zeros(Ko,Ko); Laug(1:K1,1:K1) = Lm;
SigmaP = blkdiag(Laug, eye(K2));        % K x K

% Risk prices (only current macro 2x2; zeros elsewhere)
lam0_full = [P.lambda0(1:K1); zeros(Ko-K1,1); P.lambda0(K1+1:end)]; % Kx1
lam1_MM_full = zeros(Ko,Ko); lam1_MM_full(1:K1,1:K1) = P.lambda1_MM;
lambda1 = blkdiag(lam1_MM_full, P.lambda1_LL);

% Short-rate loadings: current macro only
delta1 = [fixed.delta11; zeros(Ko-K1,1); P.delta12];   % K x 1

% Tilt to Q
muQ  = muP  - SigmaP*lam0_full;
PhiQ = PhiP - SigmaP*lambda1;

% Affine loadings
Nmax = max(data.nList);
[Aall, Ball] = riccati_affine(muQ, PhiQ, SigmaP, fixed.delta0, delta1, Nmax);

% Measurement pieces
A      = zeros(J,1);
B_Maug = zeros(J, Ko);
B_L    = zeros(J, K2);
for j=1:J
    n = data.nList(j);
    Bn = Ball(:,n);
    A(j)        = Aall(n);
    B_Maug(j,:) = Bn(1:Ko).';
    B_L(j,:)    = Bn(Ko+1:end).';
end

Ry = diag( data.v_floor(:) + exp(P.logRy_diag(:)) );

% Kalman over 3 latents
I3 = eye(3);
s_pred = zeros(3,1);  P_pred = eye(3);
nll = 0;

for t=1:To
    % Build companion macro state Xo_t = [m_t; m_{t-1}; ...; m_{t-p+1}]
    t0 = t + p;                                   % original time index
    Xo_t = zeros(Ko,1);
    for l=0:p-1, Xo_t((l*K1)+(1:K1)) = data.M_full(t0-l,:).'; end

    ybar = A./data.nList + (B_Maug./data.nList)*Xo_t;
    H    =  B_L./data.nList;
    ytil = data.Y(t,:).' - ybar;

    S = H*P_pred*H.' + Ry;
    [L,pd] = chol(S,'lower');
    if pd>0
        jitter=1e-8;
        while pd>0 && jitter<=1e-2
            S=S+jitter*eye(J); [L,pd]=chol(S,'lower'); jitter=jitter*10;
        end
        if pd>0, nll=1e12; return; end
    end

    innov = ytil - H*s_pred;
    alpha = L \ innov;
    nll = nll + 0.5*( 2*sum(log(diag(L))) + alpha.'*alpha + J*log(2*pi) );

    K = (P_pred*H.')/(L.'); K = K/L;
    s_upd = s_pred + K*innov;
    P_upd = (I3-K*H)*P_pred*(I3-K*H).' + K*Ry*K.';

    s_pred = P.RHO*s_upd;
    P_pred = P.RHO*P_upd*P.RHO.' + eye(3);
end
end

% ======================================================================
function [Ssmooth, Yhat, meas_std_bp] = smooth_and_fit_p(fixed, params, data)
% RTS smoother for 3 latents with macro companion state in measurement
K1 = data.K1;  Ko = data.Ko;  p = data.p;  J = data.J;  To = data.To;  K2 = 3;

PhiP = blkdiag(fixed.Phi_o, params.RHO);
muP  = [fixed.mu_o; zeros(K2,1)];
Lm   = chol(fixed.SigmaM,'lower'); Laug = zeros(Ko,Ko); Laug(1:K1,1:K1)=Lm;
SigmaP = blkdiag(Laug, eye(K2));

lam0_full    = [params.lambda0(1:K1); zeros(Ko-K1,1); params.lambda0(K1+1:end)];
lam1_MM_full = zeros(Ko,Ko); lam1_MM_full(1:K1,1:K1) = params.lambda1_MM;
lambda1      = blkdiag(lam1_MM_full, params.lambda1_LL);

muQ  = muP  - SigmaP*lam0_full;
PhiQ = PhiP - SigmaP*lambda1;

delta1 = [fixed.delta11; zeros(Ko-K1,1); params.delta12];

Nmax = max(data.nList);
[Aall, Ball] = riccati_affine(muQ, PhiQ, SigmaP, fixed.delta0, delta1, Nmax);

A      = zeros(J,1);
B_Maug = zeros(J, Ko);   B_L = zeros(J, K2);
for j=1:J
    n = data.nList(j);
    Bn = Ball(:,n);
    A(j)        = Aall(n);
    B_Maug(j,:) = Bn(1:Ko).';
    B_L(j,:)    = Bn(Ko+1:end).';
end
Ry = diag( data.v_floor(:) + exp(params.logRy_diag(:)) );

% Forward KF store
I3 = eye(3);
s_pred = zeros(3,1); P_pred = eye(3);
s_upd  = zeros(3,To); P_updA = zeros(3,3,To);
P_predA= zeros(3,3,To);
H_t = zeros(J,3,To);  ybar_t = zeros(J,To);

for t=1:To
    t0 = t + p;
    Xo_t = zeros(Ko,1);
    for l=0:p-1, Xo_t((l*K1)+(1:K1)) = data.M_full(t0-l,:).'; end

    ybar = A./data.nList + (B_Maug./data.nList)*Xo_t;
    H    =  B_L./data.nList;
    ytil = data.Y(t,:).' - ybar;

    S = H*P_pred*H.' + Ry;
    [L,pd] = chol(S,'lower');
    if pd>0
        jitter=1e-8;
        while pd>0 && jitter<=1e-2
            S=S+jitter*eye(J); [L,pd]=chol(S,'lower'); jitter=jitter*10;
        end
        if pd>0, error('S not PD in smoother'); end
    end
    K = (P_pred*H.')/(L.'); K = K/L;

    supd = s_pred + K*(ytil - H*s_pred);
    Pupd = (I3-K*H)*P_pred*(I3-K*H).' + K*Ry*K.';

    s_upd(:,t) = supd;  P_updA(:,:,t) = Pupd;  P_predA(:,:,t) = P_pred;
    H_t(:,:,t) = H;     ybar_t(:,t)   = ybar;

    s_pred = params.RHO*supd;
    P_pred = params.RHO*Pupd*params.RHO.' + eye(3);
end

% RTS
Ssmooth = zeros(3,To);  Ssmooth(:,To) = s_upd(:,To);
P_smooth = P_updA(:,:,To);
for t=To-1:-1:1
    Pupd = P_updA(:,:,t); Ppred_next = P_predA(:,:,t+1);
    C = Pupd * params.RHO.' / Ppred_next;
    Ssmooth(:,t) = s_upd(:,t) + C*(Ssmooth(:,t+1) - params.RHO*s_upd(:,t));
    P_smooth     = Pupd + C*(P_smooth - Ppred_next)*C.';
end

% Fitted yields
Yhat = zeros(To,J);
for t=1:To
    Yhat(t,:) = ( ybar_t(:,t) + H_t(:,:,t)*Ssmooth(:,t) ).';
end
meas_std_bp = sqrt(data.v_floor(:) + exp(params.logRy_diag(:))).' * 120000;
end

% ======================================================================
function [Aall, Ball] = riccati_affine(muQ, PhiQ, SigmaP, delta0, delta1, Nmax)
K = size(PhiQ,1);
A = zeros(Nmax,1);
B = zeros(K,Nmax);
SS = SigmaP*SigmaP.';
for k = 1:Nmax-1
    Bk = B(:,k);
    A(k+1)   = A(k) + delta0 + Bk.'*muQ + 0.5*(Bk.'*(SS*Bk));
    B(:,k+1) = PhiQ.'*Bk + delta1;
end
Aall = A;  Ball = B;
end

% ======================================================================
function theta0 = init_guess_generic(J)
% 6 (RHO lt) + 5 (lambda0) + 4 (lambda1_MM) + 9 (lambda1_LL) + 3 (delta12) + J (logRy)
target = [0.95; 0.90; 0.95];
rho0 = [atanh(target(1)); 0.02; atanh(target(2)); 0.01; 0.02; atanh(target(3))];
lambda0_0  = zeros(5,1);
lambda1MM0 = 0.015*eye(2);
lambda1LL0 = 0.015*eye(3);
delta12_0  = [0.04; 0.00; 0.04];
v15 = ((15e-4)/12)^2; logRy0 = log(max(v15,1e-10))*ones(J,1);
theta0 = [rho0; lambda0_0; lambda1MM0(:); lambda1LL0(:); delta12_0; logRy0];
end

function P = unpack_generic(theta, J)
i=0; rv = theta(i+(1:6)); i=i+6;
d1=tanh(rv(1)); d2=tanh(rv(3)); d3=tanh(rv(6));
P.RHO = [ d1 0 0; rv(2) d2 0; rv(4) rv(5) d3 ];
P.lambda0    = theta(i+(1:5)); i=i+5;
P.lambda1_MM = reshape(theta(i+(1:4)), 2,2); i=i+4;
P.lambda1_LL = reshape(theta(i+(1:9)), 3,3); i=i+9;
P.delta12    = theta(i+(1:3));  i=i+3;
P.logRy_diag = theta(i+(1:J));  i=i+J;
end

% ======================================================================
function floor_bp = auto_floors_bp(nList)
nList = nList(:)';  J = numel(nList);
floor_bp = 15*ones(1,J);
if any(nList<=6), floor_bp(nList<=6) = max(floor_bp(nList<=6),18); end
if any(nList<=3), floor_bp(nList<=3) = max(floor_bp(nList<=3),20); end
floor_bp(nList>=84 & nList<120)  = max(floor_bp(nList>=84 & nList<120),20);
floor_bp(nList>=120 & nList<168) = max(floor_bp(nList>=120 & nList<168),22);
floor_bp(nList>=168)             = max(floor_bp(nList>=168),25);
floor_bp = floor_bp(:);
end
