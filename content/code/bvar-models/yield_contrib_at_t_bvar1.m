function D = yield_contrib_at_t_bvar1(t, L, M, Ssmooth)
% Decomposition at fitted time index t (1..T):
% yhat_t(n) = A_n/n + (B_M,n/n)*m_t + (B_L,n/n)*s_t

    m = M(t,:).';          % 2x1 (observed macro)
    s = Ssmooth(:,t);      % 3x1 (smoothed latent)

    intercept = L.A_over_n;                  % J×1
    macro     = L.B_M_over_n * m;            % J×1
    latent    = L.B_L_over_n * s;            % J×1
    total     = intercept + macro + latent;  % J×1

    D = struct('nList',L.nList, ...
               'intercept',intercept, ...
               'macro',macro, ...
               'latent',latent, ...
               'total',total);
end
