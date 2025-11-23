
function plot_contrib_timeseries_bvar1(nSel, L, M, Ssmooth)
    [~, jSel] = min(abs(L.nList - nSel));
    T = size(Ssmooth,2);

    intercept_ts = L.A_over_n(jSel) * ones(T,1);
    macro_ts     = (L.B_M_over_n(jSel,:) * M.').';        % T×1
    latent_ts    = (L.B_L_over_n(jSel,:) * Ssmooth).';    % T×1

    figure('Name',sprintf('Contributions over time — %dm', L.nList(jSel)));
    plot(intercept_ts,'LineWidth',1.3); hold on;
    plot(macro_ts,'LineWidth',1.3);
    plot(latent_ts,'LineWidth',1.3);
    grid on; xlabel('t'); ylabel('Yield (monthly decimals)');
    title(sprintf('Decomposition at %d months', L.nList(jSel)));
    legend('Intercept (A/n)','Observable macro','Latent','Location','best');
end
