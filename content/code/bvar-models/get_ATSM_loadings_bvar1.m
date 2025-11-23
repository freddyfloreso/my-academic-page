function L = get_ATSM_loadings_bvar1(fixed, params, nList)
% get_ATSM_loadings_bvar1  Loadings for your BVAR(1) + 3 latent ATSM.
% Returns A_n, B_M,n (2x), B_L,n (3x), and their /n versions used in measurement.

    nList = nList(:);  J = numel(nList);

    % ----- P→Q tilt consistent with your code -----
    PhiP = blkdiag(fixed.PhiMM, params.RHO);
    muP  = [fixed.muM; zeros(3,1)];

    % SigmaP is the "square-root" used in Riccati (macro from chol(SigmaM), latent = I)
    Lm = chol_or_jitter_bvar(fixed.SigmaM);
    SigmaP = blkdiag(Lm, eye(3));

    lambda1 = blkdiag(params.lambda1_MM, params.lambda1_LL);
    muQ  = muP  - SigmaP*params.lambda0;
    PhiQ = PhiP - SigmaP*lambda1;

    delta1 = [fixed.delta11; params.delta12];

    % ----- Riccati to max maturity -----
    Nmax = max(nList);
    [Aall, Ball] = riccati_affine(muQ, PhiQ, SigmaP, fixed.delta0, delta1, Nmax);

    % ----- Slice per tenor -----
    A  = zeros(J,1);
    B_M= zeros(J,2);
    B_L= zeros(J,3);
    for j=1:J
        n = nList(j);
        Bn = Ball(:,n);        % 5x1
        A(j)      = Aall(n);
        B_M(j,:)  = Bn(1:2).';
        B_L(j,:)  = Bn(3:5).';
    end

    % Pack (and divide by n for measurement)
    L = struct();
    L.nList        = nList;
    L.A            = A;
    L.B_M          = B_M;
    L.B_L          = B_L;
    L.A_over_n     = A ./ nList;
    L.B_M_over_n   = B_M ./ nList;
    L.B_L_over_n   = B_L ./ nList;
    L.labels_macros= {'Inflation','RealActivity'};
    L.labels_latent= {'L1','L2','L3'};
end

function L = chol_or_jitter_bvar(S)
    [L,p] = chol(S,'lower');
    if p>0
        jitter = 1e-8; Sig = S;
        while p>0 && jitter <= 1e-2
            Sig = Sig + jitter*eye(size(S));
            [L,p] = chol(Sig,'lower');
            jitter = jitter*10;
        end
        if p>0, error('SigmaM not PD after jitter.'); end
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