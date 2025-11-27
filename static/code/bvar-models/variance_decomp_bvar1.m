function VD = variance_decomp_bvar1(fixed, params, nList, floors_bp)
% variance_decomp_bvar1  Unconditional variance split: macro | latent | meas
% Returns variances (monthly decimals^2), shares, and "annualized bp" stds.

    nList = nList(:);  J = numel(nList);

    % Loadings (consistent with your estimated Q)
    L = get_ATSM_loadings_bvar1(fixed, params, nList);

    % Unconditional state variances under P:
    VM = solve_dlyap_stable(fixed.PhiMM, fixed.SigmaM);  % 2x2
    VL = solve_dlyap_stable(params.RHO, eye(3));         % 3x3

    % Measurement variances used in your likelihood (monthly decimals^2)
    v_floor = (floors_bp(:)*1e-4/12).^2;                % J×1
    v_meas  = v_floor + exp(params.logRy_diag(:));      % J×1

    var_macro  = zeros(J,1);
    var_latent = zeros(J,1);
    for j=1:J
        bM = L.B_M_over_n(j,:);     % 1x2
        bL = L.B_L_over_n(j,:);     % 1x3
        var_macro(j)  = bM * VM * bM.';
        var_latent(j) = bL * VL * bL.';
    end
    var_meas  = v_meas;
    var_total = var_macro + var_latent + var_meas;

    % Shares
    share_macro  = var_macro  ./ var_total;
    share_latent = var_latent ./ var_total;
    share_meas   = var_meas   ./ var_total;

    % Std in "annualized bp" (match your rmse convention)
    sd_total_bp  = sqrt(var_total)  * 120000;
    sd_macro_bp  = sqrt(var_macro)  * 120000;
    sd_latent_bp = sqrt(var_latent) * 120000;
    sd_meas_bp   = sqrt(var_meas)   * 120000;

    % Table for convenience
    Tbl = table(nList, ...
        sd_total_bp, sd_macro_bp, sd_latent_bp, sd_meas_bp, ...
        share_macro, share_latent, share_meas, ...
        'VariableNames', {'nMonths', ...
          'sd_total_bp','sd_macro_bp','sd_latent_bp','sd_meas_bp', ...
          'share_macro','share_latent','share_meas'});

    VD = struct('nList',nList, ...
                'var_macro',var_macro, 'var_latent',var_latent, 'var_meas',var_meas, ...
                'var_total',var_total, ...
                'share_macro',share_macro, 'share_latent',share_latent, 'share_meas',share_meas, ...
                'sd_total_bp',sd_total_bp, 'sd_macro_bp',sd_macro_bp, ...
                'sd_latent_bp',sd_latent_bp, 'sd_meas_bp',sd_meas_bp, ...
                'table',Tbl);
end

function X = solve_dlyap_stable(A,Q)
% Stable discrete Lyapunov: X = A*X*A' + Q  (unique if ρ(A)<1)
    try
        X = dlyap(A, Q);  % MATLAB function
    catch
        % Fallback via vectorization: vec(X) = (I - kron(A,A))^{-1} vec(Q)
        I = eye(size(A,1)^2);
        K = I - kron(A, A);
        x = K \ Q(:);
        X = reshape(x, size(A));
    end
end
