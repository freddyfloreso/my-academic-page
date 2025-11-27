function Phi_orth = var_irf_chol(EstMdl, H, X_for_cov)
% VAR_IRF_CHOL  k×k×(H+1) orthogonalized IRFs for a fitted varm model.
% Inputs:
%   EstMdl     : fitted varm(k,p)
%   H          : horizons (e.g., 48). Returns h=0..H (H+1 slices)
%   X_for_cov  : (optional) T×k data used for infer() if Covariance is empty
% Output:
%   Phi_orth   : k×k×(H+1) IRFs to orthonormal (Cholesky) shocks

    % ---- Extract A1..Ap (fill empties with zeros) ----
    A = EstMdl.AR;                         % 1×p cell
    p = numel(A);
    ix = find(~cellfun(@isempty, A), 1);
    if isempty(ix), error('No AR blocks in model.'); end
    k = size(A{ix},1);
    for L = 1:p
        if isempty(A{L}), A{L} = zeros(k); end
    end

    % ---- Reduced-form IRFs Phi_h via recursion ----
    Phi = zeros(k,k,H+1);
    Phi(:,:,1) = eye(k);                   % h = 0
    for h = 1:H
        S = zeros(k);
        maxL = min(h,p);
        for L = 1:maxL
            S = S + A{L} * Phi(:,:,h-L+1); % note: Phi(:,:,h-L) with 0-based h
        end
        Phi(:,:,h+1) = S;
    end

    % ---- Innovation covariance & Cholesky ----
    if isprop(EstMdl,'Covariance') && ~isempty(EstMdl.Covariance)
        Sigma = EstMdl.Covariance;
    else
        if nargin < 3 || isempty(X_for_cov)
            error('Provide X_for_cov (T×k) for infer() when Covariance is empty.');
        end
        E = infer(EstMdl, X_for_cov);
        Sigma = cov(E,1);                  % MLE covariance
    end
    P = chol(Sigma,'lower');               % Sigma = P * P'

    % ---- Orthogonalize: each horizon multiplied by P ----
    Phi_orth = zeros(size(Phi));
    for h = 1:(H+1)
        Phi_orth(:,:,h) = Phi(:,:,h) * P;
    end
end
