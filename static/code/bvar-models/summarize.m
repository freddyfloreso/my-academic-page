function S = summarize(out, nList)
  T = size(out.fit.Yhat,1);
  k = 27 + numel(nList);              % params count in Step-2: 27 + J
  NLL = out.step2.nll;
  S.rmse  = out.fit.rmse_bp(:).';
  S.meas  = out.fit.meas_std_bp(:).';
  S.ratio = S.rmse ./ max(S.meas,1e-9);
  S.AIC   = 2*k + 2*NLL;
  S.BIC   = k*log(T) + 2*NLL;
end