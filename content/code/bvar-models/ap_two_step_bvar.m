function out = ap_two_step_bvar(Y_mon, M, nList, hyper, floor_bp_in)
% ap_two_step_bvar  Two-step ATSM with Step-1 Minnesota BVAR(1) and Step-2 MLE (latent)
%
% USAGE:
%   out = ap_two_step_bvar(Y_mon, M, nList)
%   out = ap_two_step_bvar(Y_mon, M, nList, hyper)
%   out = ap_two_step_bvar(Y_mon, M, nList, hyper, floor_bp_in)
%
% INPUTS:
%   Y_mon      : T x J yields in monthly decimals (annual% / 100 / 12)
%   M          : T x 2 macro factors (z-scored): [inflation, real activity]
%   nList      : 1 x J maturities in months (e.g., [3 12 24 84 120 168])
%   hyper      : (optional) struct for Minnesota prior + short-rate ridge:
%                .lam1=0.25, .lam3=0.5, .lam4=0.1, .d=1, .priorRW=true, .lam_sr=0
%   floor_bp_in: (optional) 1 x J vector of measurement std floors in bp/year
%
% OUTPUT (key fields):
%   out.step1.fixed     : struct with muM, PhiMM, SigmaM, delta0, delta11
%   out.step2.params    : struct with RHO, lambda0, lambda1_MM, lambda1_LL, delta12, logRy_diag
%   out.step2.nll       : scalar negative log-likelihood at optimum
%   out.fit.Yhat        : T x J fitted yields (monthly decimals)
%   out.fit.rmse_bp     : 1 x J RMSE in annualized bp
%   out.fit.meas_std_bp : 1 x J estimated measurement std in annualized bp (incl floor)
%   out.latent.smooth   : 3 x T smoothed latent factors
%   out.meta.floors_bp  : 1 x J floors used (bp/year)
%   out.meta.nList      : maturities (months)

% --------------------------- Input checks ---------------------------
[T,J] = size(Y_mon);
assert(size(M,1)==T && size(M,2)==2, 'M must be T x 2');
assert(numel(nList)==J, 'nList must have length = number of columns in Y_mon');
nList = nList(:)';                         % row

% --------------------------- Hyper defaults ------------------------
if nargin < 4 || isempty(hyper), hyper = struct(); end
def = struct('lam1',0.25,'lam3',0.5,'lam4',0.1,'d',1,'priorRW',true,'lam_sr',0);
fns = fieldnames(def);
for k=1:numel(fns)
    if ~isfield(hyper,fns{k}), hyper.(fns{k}) = def.(fns{k}); end
end

% --------------------------- STEP 1: BVAR(1) -----------------------
[muM, PhiMM, SigmaM] = bvar_minnesota(M, struct( ...
    'lam1', hyper.lam1, 'lam3', hyper.lam3, 'lam4', hyper.lam4, ...
    'd', hyper.d, 'priorRW', hyper.priorRW));

% Short rate on macro only: r_t = delta0 + delta11' m_t + u_t  (optional ridge)
r = Y_mon(:,1);  Xr = [ones(T,1) M];
if hyper.lam_sr > 0
    L = hyper.lam_sr * eye(size(Xr,2)); L(1,1)=0; % do not penalize intercept
    b = (Xr.'*Xr + L) \ (Xr.'*r);
else
    b = Xr \ r;
end
delta0  = b(1);
delta11 = b(2:3);

fixed.muM    = muM;           % 2x1
fixed.PhiMM  = PhiMM;         % 2x2
fixed.SigmaM = SigmaM;        % 2x2
fixed.delta0 = delta0;        % monthly
fixed.delta11= delta11;       % 2x1

% --------------------------- STEP 2: Data + floors -----------------
data.Y      = Y_mon;
data.M      = M;
data.nList  = nList(:);   % Jx1
data.T      = T;
data.J      = J;

if nargin >= 5 && ~isempty(floor_bp_in)
    assert(numel(floor_bp_in)==J, 'floor_bp_in must have length J');
    floor_bp = floor_bp_in(:)';                 % row
else
    floor_bp = auto_floors_bp(nList);          % sensible defaults by tenor
end
data.v_floor = (floor_bp*1e-4/12).^2;          % Jx1 monthly-decimal variances

% --------------------------- Initialization -----------------------
theta0 = init_guess(J);                         % (27+J) x 1
% Start measurement variances on the floor (last J entries)
theta0(end-J+1:end) = log(max(data.v_floor(:), 1e-10));

% Sanity: expected parameter count = 27 + J
Nexp = 27 + J;
assert(numel(theta0)==Nexp, 'theta0 length %d ≠ 27+J (%d)', numel(theta0), Nexp);

% --------------------------- Optimization -------------------------
obj = @(th) negloglik_latent(th, data, fixed);

opts = optimoptions('fminunc', ...
    'Display','iter', 'Algorithm','quasi-newton', ...
    'FiniteDifferenceStepSize',1e-4, ...
    'MaxIterations', 2000, 'MaxFunctionEvaluations', 5e5, ...
    'OptimalityTolerance',1e-6, 'StepTolerance',1e-8, 'FunctionTolerance',1e-8);

[theta_hat, nll, exitflag, ~] = fminunc(obj, theta0, opts);
params = unpack(theta_hat, J);   % make J-aware

% --------------------------- Smoother & fitted yields --------------
[S, Yhat, meas_std_bp] = smooth_and_fit(fixed, params, data);

% Errors & metrics
err = Y_mon - Yhat;                                  % monthly decimals
rmse_ann_bp  = sqrt(mean(err.^2, 1)) * 120000;       % annualized bp

% --------------------------- Pack output ---------------------------
out.step1.fixed      = fixed;
out.step2.params     = params;
out.step2.nll        = nll;
out.step2.exitflag   = exitflag;
out.fit.Yhat         = Yhat;
out.fit.rmse_bp      = rmse_ann_bp;
out.fit.meas_std_bp  = meas_std_bp;
out.latent.smooth    = S;
out.meta.floors_bp   = floor_bp;
out.meta.nList       = nList;

end % ===== end main =====


% ===================================================================
function [muM, PhiMM, SigmaM, Bpost] = bvar_minnesota(M, hyper)
% 2-variable BVAR(1), Minnesota prior (no toolbox). Columns of M are z-scored.

if nargin < 2 || isempty(hyper)
    hyper = struct('lam1',0.30, 'lam3',0.50, 'lam4',0.10, 'd',1, 'priorRW',true);
else
    def = struct('lam1',0.30, 'lam3',0.50, 'lam4',0.10, 'd',1, 'priorRW',true);
    f = fieldnames(def);
    for k=1:numel(f)
        if ~isfield(hyper,f{k}), hyper.(f{k}) = def.(f{k}); end
    end
end

[T,k] = size(M);  assert(k==2 && T>=3, 'M must be T x 2 with T>=3');

Mlag = M(1:end-1,:);           % (T-1) x 2
Mnow = M(2:end,:);             % (T-1) x 2
X    = [ones(T-1,1) Mlag];     % (T-1) x 3 [const, m1_{t-1}, m2_{t-1}]
q    = size(X,2);

% Scale (σ_i) from univariate AR(1) per series
sigma = zeros(2,1);
for j=1:2
    y  = Mnow(:,j);
    Z  = [ones(T-1,1) Mlag(:,j)];
    bh = Z\y;  eh = y - Z*bh;
    sigma(j) = max(std(eh), 1e-6);
end

% Prior means (RW on own-lag)
B0 = zeros(q,2);
if hyper.priorRW
    B0(2,1) = 1;  % eq1 on lag var1
    B0(3,2) = 1;  % eq2 on lag var2
end

% Prior variances (diagonal per equation)
V0 = zeros(q,q,2);
for i=1:2
    V = zeros(q,1);
    V(1) = (hyper.lam1^2) * (hyper.lam4^2) * (sigma(i)^2); % constant
    for j=1:2
        v = (hyper.lam1^2) * (sigma(i)^2 / sigma(j)^2) / (1^(2*hyper.d));
        if j~=i, v = v * (hyper.lam3^2); end
        V(1+j) = max(v,1e-12);
    end
    V0(:,:,i) = diag(V);
end

% Posterior mean (ridge) per equation
XtX = X.'*X;
Bpost = zeros(q,2);
for i=1:2
    Vi_inv = inv(V0(:,:,i));
    lhs = XtX + Vi_inv;
    rhs = (X.'*Mnow(:,i)) + Vi_inv*B0(:,i);
    [L,p] = chol(lhs,'lower');
    if p==0
        Bpost(:,i) = L'\(L\rhs);
    else
        Bpost(:,i) = lhs \ rhs;
    end
end

muM   = Bpost(1,:).';          % 2x1
PhiMM = Bpost(2:end,:).';      % 2x2

E  = Mnow - X*Bpost;
df = max((T-1)-q, 1);
SigmaM = (E.'*E) / df;

% Gentle stabilization if needed
eigPhi = abs(eig(PhiMM));
if any(eigPhi >= 1)
    s = 0.98 / max(eigPhi);
    PhiMM = PhiMM * s;
    mbar = mean(M,1).';
    muM  = (eye(2) - PhiMM) * mbar;
end
end


% ===================================================================
function nll = negloglik_latent(theta, data, fixed)
% Negative log-likelihood for latent-only state with observed macro
% State s_t (3x1): latent block
% Measurement: y_t(j) = A_n/n + (B_M/n) m_t + (B_L/n) s_t + eps

J = data.J;   T = data.T;  nList = data.nList;

% Unpack (J-aware)
P = unpack(theta, J);
% Guard
J_theta = numel(theta) - 27;  % 6+5+4+9+3
assert(J_theta==J, 'Theta implies J=%d but data.J=%d', J_theta, J);

% P-measure blocks
PhiP = blkdiag(fixed.PhiMM, P.RHO);          % 5x5
muP  = [fixed.muM; zeros(3,1)];              % 5x1

% Robust lower-triangular root for SigmaM
[Lm, pm] = chol(fixed.SigmaM, 'lower');
if pm>0
    jitter = 1e-8; Sig = fixed.SigmaM;
    while pm>0 && jitter<=1e-2
        Sig = Sig + jitter*eye(2);
        [Lm, pm] = chol(Sig,'lower');
        jitter = jitter*10;
    end
    if pm>0, nll = 1e12; return; end
end
SigmaP = blkdiag(Lm, eye(3));   % lower-tri "square-root" of Σ

% Risk prices
lambda0      = P.lambda0;         % 5x1
lambda1_MM   = P.lambda1_MM;      % 2x2
lambda1_LL   = P.lambda1_LL;      % 3x3
lambda1      = blkdiag(lambda1_MM, lambda1_LL);

% Short rate
delta0  = fixed.delta0;
delta11 = fixed.delta11;
delta12 = P.delta12;
delta1  = [delta11; delta12];     % 5x1

% Q-tilt
muQ  = muP  - SigmaP*lambda0;
PhiQ = PhiP - SigmaP*lambda1;

% Affine A,B up to max maturity
Nmax = max(nList);
[Aall, Ball] = riccati_affine(muQ, PhiQ, SigmaP, delta0, delta1, Nmax);

% Build measurement pieces
A  = zeros(J,1); B_M = zeros(J,2); B_L = zeros(J,3);
for j=1:J
    n = nList(j);
    Bn = Ball(:,n);         % 5x1
    A(j)      = Aall(n);
    B_M(j,:)  = Bn(1:2).';
    B_L(j,:)  = Bn(3:5).';
end

% Measurement covariance with tenor-specific floor
Ry = diag( data.v_floor(:) + exp(P.logRy_diag(:)) );   % JxJ

% Kalman filter (Cholesky + Joseph)
I3 = eye(3);
s_pred = zeros(3,1);
P_pred = eye(3);
nll = 0;

for t = 1:T
    ybar = A./nList + (B_M./nList) * data.M(t,:).';   % Jx1
    H    =  B_L./nList;                                % Jx3
    ytil = data.Y(t,:).' - ybar;                       % Jx1

    S = H*P_pred*H.' + Ry;
    [L,p] = chol(S,'lower');
    if p>0
        jitter = 1e-8;
        while p>0 && jitter<=1e-2
            S = S + jitter*eye(J);
            [L,p] = chol(S,'lower');
            jitter = jitter*10;
        end
        if p>0, nll = 1e12; return; end
    end

    innov = ytil - H*s_pred;
    alpha = L \ innov;
    nll = nll + 0.5*( 2*sum(log(diag(L))) + alpha.'*alpha + J*log(2*pi) );

    K = (P_pred*H.') / (L.');   K = K / L;           % chol-solve for S^{-1}

    s_upd = s_pred + K*innov;
    P_upd = (I3 - K*H)*P_pred*(I3 - K*H).' + K*Ry*K.';  % Joseph form

    s_pred = P.RHO * s_upd;
    P_pred = P.RHO * P_upd * P.RHO.' + eye(3);         % Q=I_3
end
end


% ===================================================================
function [Aall, Ball] = riccati_affine(muQ, PhiQ, SigmaP, delta0, delta1, Nmax)
% Discrete-time Gaussian ATSM:
% X_{t+1} = muQ + PhiQ X_t + SigmaP * eps_{t+1}, eps ~ N(0, I)
% r_t     = delta0 + delta1' X_t
% Price P_t(n) = exp(A_n + B_n' X_t); yields y_t(n) = -(A_n + B_n' X_t)/n (signs handled in caller)
% Recursions (Ang–Piazzesi style):
%   A_{k+1} = A_k + delta0 + B_k' muQ + 0.5 * B_k' (SigmaP*SigmaP') B_k
%   B_{k+1} = PhiQ' B_k + delta1
% with A_0=0, B_0=0.

d = size(muQ,1);
A = zeros(Nmax,1);
B = zeros(d, Nmax);     % each column k is B_k

SS = SigmaP*SigmaP.';   % covariance under Q

for k = 1:Nmax-1
    Bk = B(:,k);
    A(k+1)   = A(k) + delta0 + Bk.'*muQ + 0.5*(Bk.'*(SS*Bk));
    B(:,k+1) = PhiQ.'*Bk + delta1;
end

% Our yield formula in KF uses +A_n/n + (B/n)'X_t (no minus), so we return as-is.
Aall  = A;
Ball  = B;
end


% ===================================================================
function theta0 = init_guess(J)
% 6 (RHO lower-tri) + 5 (lambda0) + 4 (lambda1_MM) + 9 (lambda1_LL) + 3 (delta12) + J (logRy)

% Target latent persistence (actual diagonal after tanh)
target = [0.96; 0.90; 0.97];
rho0 = [atanh(target(1)); 0.02; atanh(target(2)); 0.01; 0.02; atanh(target(3))];

lambda0_0  = zeros(5,1);
lambda1MM0 = 0.012*eye(2);
lambda1LL0 = 0.018*eye(3)+0.006*ones(3);
delta12_0  = [0.06; 0.02; 0.05];

% Generic floor ~15 bp/yr (will be overwritten in main with tenor floors)
v15 = ((15e-4)/12)^2;
logRy0 = log(max(v15,1e-10)) * ones(J,1);

theta0 = [rho0; lambda0_0; lambda1MM0(:); lambda1LL0(:); delta12_0; logRy0];
end


% ===================================================================
function P = unpack(theta, J)
% Unpack vector into structured params; J-aware for logRy
i = 0;
rv = theta(i+(1:6)); i = i+6;
d1 = tanh(rv(1)); d2 = tanh(rv(3)); d3 = tanh(rv(6));
P.RHO = [ d1   0    0;
          rv(2) d2  0;
          rv(4) rv(5) d3 ];

P.lambda0    = theta(i+(1:5)); i = i+5;
P.lambda1_MM = reshape(theta(i+(1:4)), 2,2); i = i+4;
P.lambda1_LL = reshape(theta(i+(1:9)), 3,3); i = i+9;
P.delta12    = theta(i+(1:3)); i = i+3;
P.logRy_diag = theta(i+(1:J)); i = i+J;

assert(i==numel(theta), 'unpack: consumed %d of %d', i, numel(theta));
end


% ===================================================================
function [Ssmooth, Yhat, meas_std_bp] = smooth_and_fit(fixed, params, data)
% RTS smoother for 3-d latent, then construct fitted yields
T = data.T;  J = data.J;  nList = data.nList;

% Rebuild Q dynamics used in likelihood
PhiP = blkdiag(fixed.PhiMM, params.RHO);
muP  = [fixed.muM; zeros(3,1)];

[Lm, pm] = chol(fixed.SigmaM,'lower');
if pm>0
    Sig = fixed.SigmaM; jitter = 1e-8;
    while pm>0 && jitter<=1e-2
        Sig = Sig + jitter*eye(2);
        [Lm, pm] = chol(Sig,'lower');
        jitter = jitter*10;
    end
    if pm>0, error('SigmaM not PD'); end
end
SigmaP = blkdiag(Lm, eye(3));

lambda1 = blkdiag(params.lambda1_MM, params.lambda1_LL);
muQ  = muP  - SigmaP*params.lambda0;
PhiQ = PhiP - SigmaP*lambda1;

Nmax = max(nList);
[Aall, Ball] = riccati_affine(muQ, PhiQ, SigmaP, fixed.delta0, [fixed.delta11; params.delta12], Nmax);

A  = zeros(J,1); B_M = zeros(J,2); B_L = zeros(J,3);
for j=1:J
    n = nList(j);
    Bn = Ball(:,n);
    A(j)      = Aall(n);
    B_M(j,:)  = Bn(1:2).';
    B_L(j,:)  = Bn(3:5).';
end
Ry = diag( data.v_floor(:) + exp(params.logRy_diag(:)) );

% Forward KF store for RTS
I3 = eye(3);
s_pred = zeros(3,1);  P_pred = eye(3);
s_upd  = zeros(3,T);   % 3 x T
P_updA = zeros(3,3,T);
P_predA= zeros(3,3,T);
H_t    = zeros(J,3,T);
ybar_t = zeros(J,T);

for t=1:T
    ybar = A./nList + (B_M./nList) * data.M(t,:).';
    H    =  B_L./nList;
    ytil = data.Y(t,:).' - ybar;

    S  = H*P_pred*H.' + Ry;
    [L,p] = chol(S,'lower');
    if p>0
        jitter = 1e-8;
        while p>0 && jitter<=1e-2
            S = S + jitter*eye(J);
            [L,p] = chol(S,'lower');
            jitter = jitter*10;
        end
        if p>0, error('S not PD in smoother'); end
    end
    K = (P_pred*H.') / (L.');  K = K / L;

    supd = s_pred + K*(ytil - H*s_pred);
    Pupd = (I3 - K*H)*P_pred*(I3 - K*H).' + K*Ry*K.';

    s_upd(:,t) = supd;
    P_updA(:,:,t) = Pupd;
    P_predA(:,:,t)= P_pred;
    H_t(:,:,t)    = H;
    ybar_t(:,t)   = ybar;

    s_pred = params.RHO * supd;
    P_pred = params.RHO * Pupd * params.RHO.' + eye(3);
end

% RTS smoother
Ssmooth = zeros(3,T);
Ssmooth(:,T) = s_upd(:,T);
P_smooth = P_updA(:,:,T);
for t=T-1:-1:1
    Pupd = P_updA(:,:,t);
    Ppred_next = P_predA(:,:,t+1);
    C = Pupd * params.RHO.' / Ppred_next;
    Ssmooth(:,t) = s_upd(:,t) + C * (Ssmooth(:,t+1) - (params.RHO*s_upd(:,t)));
    P_smooth = Pupd + C*(P_smooth - Ppred_next)*C.';
end

% Fitted yields: yhat = A/n + (B_M/n) m_t + (B_L/n) s_smooth
Yhat = zeros(T,J);
for t=1:T
    ybar = ybar_t(:,t);
    H = H_t(:,:,t);
    Yhat(t,:) = (ybar + H * Ssmooth(:,t)).';
end

% Measurement std (annual bp)
meas_std_bp = sqrt(data.v_floor(:) + exp(params.logRy_diag(:))).' * 120000;
end


% ===================================================================
function floor_bp = auto_floors_bp(nList)
% Simple end-heavier floor in bp/year given maturities (months).
nList = nList(:)';
J = numel(nList);
floor_bp = 15*ones(1,J);         % base in the belly

% Short end slightly higher
if any(nList <= 6),  floor_bp(nList<=6) = 18; end
if any(nList <= 3),  floor_bp(nList<=3) = 20; end

% Longer maturities slightly higher
floor_bp(nList>=84 & nList<120) = max(floor_bp(nList>=84 & nList<120), 20);
floor_bp(nList>=120 & nList<168)= max(floor_bp(nList>=120 & nList<168),22);
floor_bp(nList>=168)            = max(floor_bp(nList>=168),            25);
end


