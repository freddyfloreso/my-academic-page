function [Z, loc, scale] = robust_z(X, varargin)
% ROBUST_Z: column-wise robust standardization
% Z = (X - loc) ./ scale, with loc=median, scale=1.4826*MAD
% Falls back to IQR or STD if MAD is too small or NaN.
%
% Usage:
%   [Z,loc,scale] = robust_z(X);                            % fit on full X
%   [Z,loc,scale] = robust_z(X, 'BaseIdx', idx);           % fit on base window
%   [Z,loc,scale] = robust_z(X, 'Clip', 6);                % winsorize z to ±6
%
% X can be T×K; returns K-by-1 loc/scale.

  p = inputParser;
  addParameter(p,'BaseIdx',true(size(X,1),1));  % logical index for fitting window
  addParameter(p,'Clip',Inf);                   % clip z-scores
  parse(p,varargin{:});
  base = p.Results.BaseIdx;
  clip = p.Results.Clip;

  % location: median on base window
  loc = median(X(base,:), 1, 'omitnan');

  % unscaled MAD on base window
  mad_un = median(abs(X(base,:) - loc), 1, 'omitnan');
  scale  = 1.4826 .* mad_un;

  % fallback to IQR/1.349 if MAD too small/NaN
  bad = isnan(scale) | (scale < 1e-8);
  if any(bad)
      q = quantile(X(base, bad), [0.25 0.75], 1);
      iqr = q(2,:) - q(1,:);
      scale(bad) = iqr ./ 1.349;
  end

  % final fallback to STD if still bad
  bad = isnan(scale) | (scale < 1e-8);
  if any(bad)
      scale(bad) = std(X(base, bad), 0, 1, 'omitnan');
  end

  % avoid division by zero
  scale(scale < 1e-12 | isnan(scale)) = 1;

  % standardize full sample with base loc/scale
  Z = (X - loc) ./ scale;

  % optional clipping of extreme z-scores
  if isfinite(clip)
      Z = max(min(Z, clip), -clip);
  end
end
