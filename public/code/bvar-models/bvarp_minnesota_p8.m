function out = bvarp_minnesota_p8(M, p, H, hyper)
% BVAR(p) with Minnesota prior for a 2D macro block (inflation, real activity).
% Returns posterior means (mu, Phi, Sigma) and Cholesky IRFs up to horizon H,
% and (added) posterior std. errors, t-stats, p-values for coefficients.
%
% INPUTS
%   M     : T x 2 data (columns: [inflation, real activity]), ideally z-scored
%   p     : number of lags (default 8)
%   H     : IRF horizon in months (default 48)
%   hyper : struct with Minnesota hyperparameters (all optional):
%           .lam1 (overall tightness, default 0.30)
%           .lam3 (cross-tightness, default 0.50)
%           .lam4 (constant tightness, default 0.10)
%           .d    (lag decay, default 1)
%           .priorRW (true -> own lag-1 prior mean 1; others 0; default true)
%
% OUTPUT (struct "out")
%   .muM      : 2x1 intercepts (posterior mean)
%   .PhiStack : 2 x (2p) stacked lag matrices, [Phi1 | Phi2 | ... | Phip]
%   .PhiM     : 2 x 2 x p array with Phi(:,:,ℓ) the lag-ℓ coefficient matrix
%   .SigmaM   : 2x2 innovation covariance (from residuals at posterior mean)
%   .Bpost    : (1+2p) x 2 coefficient matrix [const, lag1(m1), lag1(m2), ..., lagp(m2)]
%   .Fcomp    : 2p x 2p companion matrix at posterior mean
%   .IRF      : 2 x 2 x (H+1) Cholesky IRFs (shock to j, response of i), horizon 0..H
%   --- added ---
%   .VB(:,:,i): posterior covariance of coefficients (eq i), size (1+2p)x(1+2p)
%   .seB(:,i) : posterior std. errors of coefficients (eq i)
%   .tB(:,i)  : z/t stats (coef / se)  (eq i)
%   .pB(:,i)  : two-sided p-values via normal approximation (eq i)
%   .paramNames: cell(q,1) parameter labels matching rows of Bpost
%   .tables.eq1/eq2: MATLAB tables with Param, Coef, StdError, Tstat, PValue
%
% NOTES
% - Prior scale (sigma_j) is estimated from univariate AR(p) OLS for each series.
% - Minnesota prior variance: lam1^2 * (sigma_i^2 / sigma_j^2) / (lag^(2d)),
%   multiplied by lam3^2 for cross-variable coefficients; constant has lam1^2*lam4^2*sigma_i^2.
% - IRFs are computed in companion form with a Cholesky identification.

    if nargin < 2 || isempty(p), p = 8; end
    if nargin < 3 || isempty(H), H = 48; end
    if nargin < 4, hyper = struct; end

    defaults = struct('lam1',0.30,'lam3',0.50,'lam4',0.10,'d',1,'priorRW',true);
    f = fieldnames(defaults);
    for k=1:numel(f)
        if ~isfield(hyper,f{k}), hyper.(f{k}) = defaults.(f{k}); end
    end

    [T,k] = size(M);
    assert(k==2,'M must be T x 2');
    assert(T > p, 'Need T > p observations.');

    % ---------------------------
    % Build VAR(p) regression
    % m_t = c + Phi1*m_{t-1} + ... + Phip*m_{t-p} + e_t
    % ---------------------------
    Y = M((p+1):end,:);                % (T-p) x 2
    Xlags = [];
    for lag = 1:p
        Xlags = [Xlags, M((p+1-lag):(end-lag),:)]; %#ok<AGROW>
    end
    X = [ones(T-p,1), Xlags];          % (T-p) x (1+2p)
    q = size(X,2);                     % 1 + 2p
    XtX = X.'*X;

    % ---------------------------
    % Prior scale sigmas from AR(p) per series (with constant)
    % ---------------------------
    sigma = zeros(2,1);
    for j=1:2
        yj = M((p+1):end, j);
        Xj = [ones(T-p,1)];
        for lag=1:p
            Xj = [Xj, M((p+1-lag):(end-lag), j)]; %#ok<AGROW>
        end
        bj = Xj\yj;
        ej = yj - Xj*bj;
        sigma(j) = max(std(ej), 1e-8);
    end

    % ---------------------------
    % Minnesota prior: B0 (prior mean) and V0 (diag prior variance) eq-by-eq
    % ---------------------------
    B0 = zeros(q,2);
    if hyper.priorRW
        % Own lag-1 prior mean = 1 for each equation's own variable; others 0
        % Order in X: [const, (y_{t-1}, x_{t-1}), (y_{t-2}, x_{t-2}), ..., (y_{t-p}, x_{t-p})]
        B0(1 + 1, 1) = 1; % eq1 own L1
        B0(1 + 2, 2) = 1; % eq2 own L1
    end

    V0 = zeros(q,q,2);
    for i=1:2
        V = zeros(q,1);
        % Constant:
        V(1) = (hyper.lam1^2) * (hyper.lam4^2) * (sigma(i)^2);
        % Lagged regressors:
        for ell = 1:p
            for j = 1:2
                idx = 1 + 2*(ell-1) + j;
                v = (hyper.lam1^2) * (sigma(i)^2 / sigma(j)^2) / (ell^(2*hyper.d));
                if j ~= i
                    v = v * (hyper.lam3^2);
                end
                V(idx) = v;
            end
        end
        V0(:,:,i) = diag(max(V, 1e-12));
    end

    % ---------------------------
    % Posterior mean (ridge closed form) by equation
    % ---------------------------
    Bpost = zeros(q,2);
    for i=1:2
        % inv(V0) cheaply since it's diagonal
        Vi_inv = diag(1./diag(V0(:,:,i)));
        lhs = XtX + Vi_inv;
        rhs = (X.'*Y(:,i)) + Vi_inv*B0(:,i);
        Bpost(:,i) = lhs \ rhs;
    end

    % ---------------------------
    % Residuals, covariance
    % ---------------------------
    E = Y - X*Bpost;                  % (T-p) x 2
    df = max((T-p) - q, 1);
    SigmaM = (E.'*E) / df;

    % ---------------------------
    % Posterior std. errors, t-stats, p-values (per equation)
    % Var(beta_i | Y, Sigma) = SigmaM(i,i) * (X'X + V0^{-1})^{-1}
    % ---------------------------
    out.VB = zeros(q,q,2);
    out.seB = zeros(q,2);
    out.tB  = zeros(q,2);
    out.pB  = zeros(q,2);

    for i=1:2
        Vi_inv = diag(1./diag(V0(:,:,i)));
        Ai = XtX + Vi_inv;
        % Inverse via Cholesky:
        R  = chol(Ai);                      % Ai = R'*R
        VXi = R \ (R' \ eye(q));            % (X'X + V0^{-1})^{-1}
        Vi = SigmaM(i,i) * VXi;             % posterior covariance of betas, eq i

        se = sqrt(diag(Vi));
        t  = Bpost(:,i) ./ se;
        % two-sided normal p-value without Stats TBX: p = erfc(|t| / sqrt(2))
        pval = erfc(abs(t)/sqrt(2));

        out.VB(:,:,i) = Vi;
        out.seB(:,i)  = se;
        out.tB(:,i)   = t;
        out.pB(:,i)   = pval;
    end

    % ---------------------------
    % Labels and convenience tables
    % ---------------------------
    names = cell(q,1); names{1} = 'const';
    for ell = 1:p
        names{2*(ell-1)+2} = sprintf('L%d infl', ell);
        names{2*(ell-1)+3} = sprintf('L%d real', ell);
    end
    out.paramNames = names;

    out.tables.eq1 = table(names, Bpost(:,1), out.seB(:,1), out.tB(:,1), out.pB(:,1), ...
        'VariableNames', {'Parametro','Coef','StdError','Tstat','PValue'});
    out.tables.eq2 = table(names, Bpost(:,2), out.seB(:,2), out.tB(:,2), out.pB(:,2), ...
        'VariableNames', {'Parametro','Coef','StdError','Tstat','PValue'});

    % ---------------------------
    % Read out mu and Phi (stacked and per-lag)
    % ---------------------------
    muM = Bpost(1,:).';               % 2x1
    PhiStack = zeros(2, 2*p);
    for ell=1:p
        cols = (2*(ell-1)+1):(2*ell);
        PhiStack(:, cols) = Bpost(1+cols,:).';  % 2 x 2 block for lag ell
    end
    PhiM = zeros(2,2,p);
    for ell=1:p
        cols = (2*(ell-1)+1):(2*ell);
        PhiM(:,:,ell) = PhiStack(:, cols);
    end

    % ---------------------------
    % Companion matrix for IRFs
    % ---------------------------
    Fcomp = zeros(2*p, 2*p);
    % Top block: [Phi1 ... Phip]
    Fcomp(1:2,1:(2*p)) = PhiStack;
    % Sub-identity to shift lags
    if p > 1
        Fcomp(3:(2*p), 1:(2*(p-1))) = eye(2*(p-1));
    end

    % ---------------------------
    % IRFs (Cholesky identification)
    % ---------------------------
    [Pc, pflag] = chol(SigmaM, 'lower');
    if pflag > 0
        Pc = chol(SigmaM + 1e-8*eye(2), 'lower');
    end
    Bshock = [Pc; zeros(2*(p-1), 2)];   % (2p x 2)
    IRF = zeros(2,2,H+1);
    IRF(:,:,1) = Pc;                    % h = 0
    FhB = Bshock;
    for h=1:H                           % h >= 1
        FhB = Fcomp * FhB;
        IRF(:,:,h+1) = FhB(1:2, :);
    end

    % ---------------------------
    % Pack output
    % ---------------------------
    out.muM      = muM;
    out.PhiStack = PhiStack;   % 2 x (2p)
    out.PhiM     = PhiM;       % 2 x 2 x p
    out.SigmaM   = SigmaM;
    out.Bpost    = Bpost;      % (1+2p) x 2
    out.Fcomp    = Fcomp;      % 2p x 2p
    out.IRF      = IRF;        % 2 x 2 x (H+1)
end
