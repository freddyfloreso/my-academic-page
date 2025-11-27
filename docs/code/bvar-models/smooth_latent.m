function S = smooth_latent(out, Y, M, nList)
% Smooth the 3 latent TS factors using an already-estimated two-step ATSM
% produced by ap_two_step_var_p8 (VAR(p) in Step-1).
%
% INPUTS
%   out    : struct from ap_two_step_var_p8
%   Y      : T x J yields in monthly decimals (same data you used)
%   M      : T x K1 macro (observed, z-scored)
%   nList  : 1 x J maturities in months
%
% OUTPUT
%   S.smooth_latent : 3 x To smoothed latent states (aligned with Y(p+1:end,:))
%   S.Yhat          : To x J fitted yields from smoothed states
%   S.meas_std_bp   : 1 x J measurement std (annualized, bp)
%   S.misc          : struct with A, B_Maug, B_L, and key blocks

assert(isfield(out,'step1') && isfield(out,'step2'), 'Out must come from ap_two_step_var_p8');

fixed  = out.step1.fixed;
params = out.step2.params;

if isfield(out,'meta') && isfield(out.meta,'p')
    p = out.meta.p;
else
    % Infer p from companion size if needed
    K1 = size(M,2);
    Ko = size(fixed.Phi_o,1);
    p  = Ko / K1;
    assert(abs(p-round(p))<1e-12, 'Cannot infer VAR order p.');
end

% Build data bundle exactly like ap_two_step_var_p8
T         = size(Y,1);
K1        = size(M,2);
Ko        = K1*p;
nList     = nList(:);
J         = numel(nList);
To        = T - p;

data.Y      = Y(p+1:end, :);   % aligned with Mnow
data.M_full = M;               % for building lags on the fly
data.To     = To;  data.J = J; data.nList = nList;
data.p      = p;   data.K1 = K1; data.Ko = Ko;

% Use floors from 'out' if present; otherwise rebuild defaults
if isfield(out,'meta') && isfield(out.meta,'floors_bp') && numel(out.meta.floors_bp)==J
    floor_bp = out.meta.floors_bp(:);
else
    floor_bp = auto_floors_bp(nList);
end
data.v_floor = (floor_bp*1e-4/12).^2;

% ---- run the same smoother as in ap_two_step_var_p8 (inlined here) ----
[K1_, Ko_, p_, J_, To_] = deal(data.K1, data.Ko, data.p, data.J, data.To); %#ok<ASGLU>
K2 = 3;

PhiP = blkdiag(fixed.Phi_o, params.RHO);
muP  = [fixed.mu_o; zeros(K2,1)];

% Robust chol on SigmaM and "augment" to companion dimension
[Lm, pm] = chol(fixed.SigmaM,'lower');
if pm>0
    Sig = fixed.SigmaM; jit=1e-8;
    while pm>0 && jit<=1e-2
        Sig = Sig + jit*eye(K1);
        [Lm, pm] = chol(Sig,'lower'); jit = jit*10;
    end
    assert(pm==0,'SigmaM not PD.');
end
Laug = zeros(Ko,Ko); Laug(1:K1,1:K1)=Lm;
SigmaP = blkdiag(Laug, eye(K2));

% Risk prices and short-rate (current macro only)
lam0_full    = [params.lambda0(1:K1); zeros(Ko-K1,1); params.lambda0(K1+1:end)];
lam1_MM_full = zeros(Ko,Ko); lam1_MM_full(1:K1,1:K1)=params.lambda1_MM;
lambda1      = blkdiag(lam1_MM_full, params.lambda1_LL);
muQ  = muP  - SigmaP*lam0_full;
PhiQ = PhiP - SigmaP*lambda1;
delta1 = [fixed.delta11; zeros(Ko-K1,1); params.delta12];

% Affine loadings
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

% Kalman filter (3-d latent block only)
I3 = eye(3);
s_pred = zeros(3,1); P_pred = eye(3);
s_upd  = zeros(3,To); P_updA = zeros(3,3,To); P_predA = zeros(3,3,To);
H_t = zeros(J,3,To); ybar_t = zeros(J,To);

for t=1:To
    t0 = t + p;                      % aligned with Mnow
    Xo_t = zeros(Ko,1);
    for l=0:p-1
        Xo_t((l*K1)+(1:K1)) = data.M_full(t0-l,:).';
    end

    ybar = A./data.nList + (B_Maug./data.nList)*Xo_t;
    H    =  B_L./data.nList;
    ytil = data.Y(t,:).' - ybar;

    Sinnov = H*P_pred*H.' + Ry;
    [L,pd] = chol(Sinnov,'lower');
    if pd>0
        jit=1e-8;
        while pd>0 && jit<=1e-2
            Sinnov=Sinnov+jit*eye(J);
            [L,pd]=chol(Sinnov,'lower'); jit=jit*10;
        end
        assert(pd==0,'Innovation covariance not PD in smoother.');
    end

    K = (P_pred*H.')/(L.'); K = K / L;
    supd = s_pred + K*(ytil - H*s_pred);
    Pupd = (I3-K*H)*P_pred*(I3-K*H).' + K*Ry*K.';   % Joseph

    s_upd(:,t) = supd;  P_updA(:,:,t) = Pupd;  P_predA(:,:,t) = P_pred;
    H_t(:,:,t) = H;     ybar_t(:,t)   = ybar;

    s_pred = params.RHO*supd;
    P_pred = params.RHO*Pupd*params.RHO.' + eye(3);
end

% RTS smoother
Ssmooth = zeros(3,To);  Ssmooth(:,To) = s_upd(:,To);
P_smooth = P_updA(:,:,To);
for t=To-1:-1:1
    Pupd = P_updA(:,:,t);  Ppred_next = P_predA(:,:,t+1);
    C = Pupd * params.RHO.' / Ppred_next;
    Ssmooth(:,t) = s_upd(:,t) + C*(Ssmooth(:,t+1) - params.RHO*s_upd(:,t));
    P_smooth = Pupd + C*(P_smooth - Ppred_next)*C.';
end

% Fitted yields from smoothed state
Yhat = zeros(To,J);
for t=1:To
    Yhat(t,:) = ( ybar_t(:,t) + H_t(:,:,t)*Ssmooth(:,t) ).';
end
meas_std_bp = sqrt(data.v_floor(:) + exp(params.logRy_diag(:))).' * 120000;

% Pack
S.smooth_latent = Ssmooth;
S.Yhat          = Yhat;
S.meas_std_bp   = meas_std_bp;
S.misc.A        = A;
S.misc.B_Maug   = B_Maug;
S.misc.B_L      = B_L;
S.misc.Kalman_R = Ry;
S.misc.blocks   = struct('PhiP',PhiP,'muP',muP,'SigmaP',SigmaP, ...
                         'muQ',muQ,'PhiQ',PhiQ, ...
                         'delta0',fixed.delta0,'delta1',delta1);

end

% ---- helpers (same conventions as your two-step code) ----
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
