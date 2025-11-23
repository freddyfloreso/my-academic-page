function plot_loading_curves_bvar1(L)
    n = L.nList;
    figure('Name','Loadings — Macro (observable) & Latent'); tiledlayout(2,1);

    % Macro loadings (observable)
    nexttile;
    plot(n, L.B_M_over_n(:,1),'LineWidth',1.6); hold on;
    plot(n, L.B_M_over_n(:,2),'LineWidth',1.6);
    grid on; xlabel('Maturity (months)'); ylabel('Loading per 1/n');
    title('Observable macro loadings');
    legend('Inflation','Real Activity','Location','best');

    % Latent loadings
    nexttile;
    plot(n, L.B_L_over_n(:,1),'LineWidth',1.6); hold on;
    plot(n, L.B_L_over_n(:,2),'LineWidth',1.6);
    plot(n, L.B_L_over_n(:,3),'LineWidth',1.6);
    grid on; xlabel('Maturity (months)'); ylabel('Loading per 1/n');
    title('Latent loadings');
    legend('L1','L2','L3','Location','best');
end

