function out = ap_two_step_var_p8(Y_mon, M, nList, floor_bp_in, lam_sr)
% Two-step ATSM with Step-1 OLS VAR(8) for macro and Step-2 MLE (latent).
%
% INPUTS:
%   Y_mon : T x J yields in monthly decimals
%   M     : T x 2 macro factors (z-scored): [inflation, real activity]
%   nList : 1 x J maturities in months (row or col)
%   floor_bp_in (opt): 1 x J measurement floors in bp/year
%   lam_sr (opt): ridge for short-rate-on-macro regression (default 0)

if nargin < 4, floor_bp_in = []; end
if nargin < 5 || isempty(lam_sr), lam_sr = 0; end

[T,J0] = size(Y_mon);                           %#ok<NASGU>
assert(size(M,1)==T && size(M,2)==2, 'M must be T x 2');
nList = nList(:);  J = numel(nList);

K1 = 2;               % observed macro
K2 = 3;               % latent
p  = 8;               % VAR order
To = T - p;           % <-- effective sample (targets m_{p+1},...,m_T)
assert(To >= 50, 'Not enough data after building %d lags.', p);

% --------------------------- STEP 1: OLS VAR(8) ---------------------------
% Targets: Mnow = m_{p+1}..m_T  (To x K1)
Mnow = M(p+1:end, :);
% Regressors: [1, m_{t-1},...,m_{t-p}] length = To
Xvar = ones(To, 1 + K1*p);
for l = 1:p
    % rows: (p+1-l) .. (T-l), length To
    Xvar(:, 1 + (K1*(l-1)+(1:K1))) = M((p+1-l):(T-l), :);
end

B      = Xvar \ Mnow;                   % (1+K1*p) x K1
c      = B(1,:).';                      % K1 x 1
PhiAll = B(2:end,:).';                  % K1 x (K1*p)

Evar   = Mnow - Xvar*B;
SigmaM = (Evar.'*Evar) / max(To - (1+K1*p), 1);

% Companion (Ko = K1*p)
Ko = K1*p;
Phi_o = zeros(Ko, Ko);
Phi_o(1:K1,:)                 = PhiAll;
Phi_o((K1+1):Ko,1:(Ko-K1))    = eye(Ko-K1);
mu_o = zeros(Ko,1);  mu_o(1:K1) = c;

% Short rate on CURRENT macro only, aligned with Mnow
r  = Y_mon(p+1:end,1);                  % To x 1
Xr = [ones(To,1) Mnow];                 % To x (1+K1)
if lam_sr > 0
    L = lam_sr*eye(size(Xr,2)); L(1,1)=0;
    b = (Xr.'*Xr + L) \ (Xr.'*r);
else
    b = Xr \ r;
end
delta0  = b(1);
delta11 = b(2:1+K1);

fixed.mu_o    = mu_o;
fixed.Phi_o   = Phi_o;
fixed.SigmaM  = SigmaM;
fixed.delta0  = delta0;
fixed.delta11 = delta11;

% --------------------------- STEP 2 data ----------------------------------
data.Y      = Y_mon(p+1:end, :);        % To x J  (aligned)
data.M_full = M;                        % for building lags on the fly
data.To     = To;
data.J      = J;
data.nList  = nList;
data.p      = p;
data.K1     = K1;
data.Ko     = Ko;

% Floors (bp/year) -> monthly variance
if ~isempty(floor_bp_in)
    assert(numel(floor_bp_in)==J, 'floor_bp_in must have length J');
    floor_bp = floor_bp_in(:);
else
    floor_bp = auto_floors_bp(nList);
end
data.v_floor = (floor_bp*1e-4/12).^2;

% --------------------------- Optimization ---------------------------------
theta0 = init_guess_generic(J);
theta0(end-J+1:end) = log(max(data.v_floor(:), 1e-10));

obj  = @(th) negloglik_latent_p8(th, data, fixed);
opts = optimoptions('fminunc','Display','iter','Algorithm','quasi-newton', ...
    'FiniteDifferenceStepSize',1e-4,'MaxIterations',2000, ...
    'MaxFunctionEvaluations',5e5,'OptimalityTolerance',1e-6, ...
    'StepTolerance',1e-8,'FunctionTolerance',1e-8);

[theta_hat, nll, exitflag] = fminunc(obj, theta0, opts); %#ok<ASGLU>
params = unpack_generic(theta_hat, J);

% --------------------------- Smoother & fitted ----------------------------
[S, Yhat, meas_std_bp] = smooth_and_fit_p8(fixed, params, data);

err = data.Y - Yhat;
rmse_ann_bp  = sqrt(mean(err.^2, 1)) * 120000;

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

% ======================= NEG-LOGLIK (VAR p=8) ============================
function nll = negloglik_latent_p8(theta, data, fixed)
P  = unpack_generic(theta, data.J);
K1 = data.K1;  Ko = data.Ko;  K2 = 3;  J = data.J;
To = data.To;  p  = data.p;

PhiP = blkdiag(fixed.Phi_o, P.RHO);
muP  = [fixed.mu_o; zeros(K2,1)];

Lm   = chol(fixed.SigmaM,'lower');
Laug = zeros(Ko,Ko);  Laug(1:K1,1:K1) = Lm;
SigmaP = blkdiag(Laug, eye(K2));

% Risk prices (current macro only)
lam0_full      = [P.lambda0(1:K1); zeros(Ko-K1,1); P.lambda0(K1+1:end)];
lam1_MM_full   = zeros(Ko,Ko); lam1_MM_full(1:K1,1:K1) = P.lambda1_MM;
lambda1        = blkdiag(lam1_MM_full, P.lambda1_LL);
delta1         = [fixed.delta11; zeros(Ko-K1,1); P.delta12];

muQ  = muP  - SigmaP*lam0_full;
PhiQ = PhiP - SigmaP*lambda1;

Nmax = max(data.nList);
[Aall, Ball] = riccati_affine(muQ, PhiQ, SigmaP, fixed.delta0, delta1, Nmax);

A      = zeros(J,1);
B_Maug = zeros(J, Ko);   B_L = zeros(J, K2);
for j=1:J
    n = data.nList(j);  Bn = Ball(:,n);
    A(j)        = Aall(n);
    B_Maug(j,:) = Bn(1:Ko).';
    B_L(j,:)    = Bn(Ko+1:end).';
end

Ry = diag( data.v_floor(:) + exp(P.logRy_diag(:)) );

% Kalman over latents (3-dim)
I3 = eye(3);
s_pred = zeros(3,1);  P_pred = eye(3);
nll = 0;

for t = 1:To
    t0 = t + p;                                % <-- aligned with Mnow
    Xo_t = zeros(Ko,1);
    for l=0:p-1
        Xo_t((l*K1)+(1:K1)) = data.M_full(t0-l,:).';
    end

    ybar = A./data.nList + (B_Maug./data.nList)*Xo_t;
    H    =  B_L./data.nList;
    ytil = data.Y(t,:).' - ybar;

    S  = H*P_pred*H.' + Ry;
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
    nll = nll + 0.5*(2*sum(log(diag(L))) + alpha.'*alpha + J*log(2*pi));

    K = (P_pred*H.')/(L.');  K = K/L;
    s_upd = s_pred + K*innov;
    P_upd = (I3-K*H)*P_pred*(I3-K*H).' + K*Ry*K.';   % Joseph

    s_pred = P.RHO*s_upd;
    P_pred = P.RHO*P_upd*P.RHO.' + eye(3);
end
end

% ============================== SMOOTHER ================================
function [Ssmooth, Yhat, meas_std_bp] = smooth_and_fit_p8(fixed, params, data)
K1 = data.K1;  Ko = data.Ko;  p = data.p;  J = data.J;  To = data.To;  K2 = 3;

PhiP = blkdiag(fixed.Phi_o, params.RHO);
muP  = [fixed.mu_o; zeros(K2,1)];
Lm   = chol(fixed.SigmaM,'lower'); Laug = zeros(Ko,Ko); Laug(1:K1,1:K1)=Lm;
SigmaP = blkdiag(Laug, eye(K2));
lam0_full    = [params.lambda0(1:K1); zeros(Ko-K1,1); params.lambda0(K1+1:end)];
lam1_MM_full = zeros(Ko,Ko); lam1_MM_full(1:K1,1:K1)=params.lambda1_MM;
lambda1      = blkdiag(lam1_MM_full, params.lambda1_LL);
muQ  = muP  - SigmaP*lam0_full;
PhiQ = PhiP - SigmaP*lambda1;
delta1 = [fixed.delta11; zeros(Ko-K1,1); params.delta12];

Nmax = max(data.nList);
[Aall, Ball] = riccati_affine(muQ, PhiQ, SigmaP, fixed.delta0, delta1, Nmax);

A      = zeros(J,1);
B_Maug = zeros(J, Ko);   B_L = zeros(J, K2);
for j=1:J
    n = data.nList(j);  Bn = Ball(:,n);
    A(j)        = Aall(n);
    B_Maug(j,:) = Bn(1:Ko).';
    B_L(j,:)    = Bn(Ko+1:end).';
end
Ry = diag( data.v_floor(:) + exp(params.logRy_diag(:)) );

I3 = eye(3);
s_pred = zeros(3,1); P_pred = eye(3);
s_upd  = zeros(3,To); P_updA = zeros(3,3,To); P_predA = zeros(3,3,To);
H_t = zeros(J,3,To); ybar_t = zeros(J,To);

for t=1:To
    t0 = t + p;                                % <-- aligned
    Xo_t = zeros(Ko,1);
    for l=0:p-1
        Xo_t((l*K1)+(1:K1)) = data.M_full(t0-l,:).';
    end

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
    Pupd = P_updA(:,:,t);  Ppred_next = P_predA(:,:,t+1);
    C = Pupd * params.RHO.' / Ppred_next;
    Ssmooth(:,t) = s_upd(:,t) + C*(Ssmooth(:,t+1) - params.RHO*s_upd(:,t));
    P_smooth = Pupd + C*(P_smooth - Ppred_next)*C.';
end

% Fitted yields
Yhat = zeros(To,J);
for t=1:To
    Yhat(t,:) = ( ybar_t(:,t) + H_t(:,:,t)*Ssmooth(:,t) ).';
end
meas_std_bp = sqrt(data.v_floor(:) + exp(params.logRy_diag(:))).' * 120000;
end

% ============================ Riccati =============================
function [Aall, Ball] = riccati_affine(muQ, PhiQ, SigmaP, delta0, delta1, Nmax)
K = size(PhiQ,1);  A = zeros(Nmax,1);  B = zeros(K,Nmax);
SS = SigmaP*SigmaP.';
for k = 1:Nmax-1
    Bk = B(:,k);
    A(k+1)   = A(k) + delta0 + Bk.'*muQ + 0.5*(Bk.'*(SS*Bk));
    B(:,k+1) = PhiQ.'*Bk + delta1;
end
Aall = A;  Ball = B;
end

% ===================== Pack / Unpack (same as before) =====================
function theta0 = init_guess_generic(J)
target = [0.90; 0.88; 0.85];
rho0 = [atanh(target(1)); 0.02; atanh(target(2)); 0.01; 0.02; atanh(target(3))];
lambda0_0  = zeros(5,1);
lambda1MM0 = 0.015*eye(2);
lambda1LL0 = 0.015*eye(3);
delta12_0  = [0.03; 0.00; 0.03];
v15 = ((15e-4)/12)^2; logRy0 = log(max(v15,1e-10))*ones(J,1);
theta0 = [rho0; lambda0_0; lambda1MM0(:); lambda1LL0(:); delta12_0; logRy0];
end

function P = unpack_generic(theta, J)
i = 0; rv = theta(i+(1:6)); i=i+6;
d1=tanh(rv(1)); d2=tanh(rv(3)); d3=tanh(rv(6));
P.RHO = [ d1 0  0;  rv(2) d2 0;  rv(4) rv(5) d3 ];
P.lambda0    = theta(i+(1:5));  i=i+5;
P.lambda1_MM = reshape(theta(i+(1:4)), 2,2); i=i+4;
P.lambda1_LL = reshape(theta(i+(1:9)), 3,3); i=i+9;
P.delta12    = theta(i+(1:3));  i=i+3;
P.logRy_diag = theta(i+(1:J));  i=i+J;
end

% ============================== Floors helper ======================
function floor_bp = auto_floors_bp(nList)
nList = nList(:)';  J = numel(nList);
floor_bp = 15*ones(1,J);
floor_bp(nList<=6) = max(floor_bp(nList<=6),18);
floor_bp(nList<=3) = max(floor_bp(nList<=3),20);
floor_bp(nList>=84 & nList<120)  = max(floor_bp(nList>=84 & nList<120),20);
floor_bp(nList>=120 & nList<168) = max(floor_bp(nList>=120 & nList<168),22);
floor_bp(nList>=168)             = max(floor_bp(nList>=168),25);
floor_bp = floor_bp(:);
end
