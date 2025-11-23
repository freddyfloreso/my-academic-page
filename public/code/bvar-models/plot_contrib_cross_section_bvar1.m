function plot_contrib_cross_section_bvar1(D)
    X = [D.intercept, D.macro, D.latent];  % J×3
    figure('Name','Yield decomposition across maturities');
    bar(D.nList, X, 'stacked', 'BarWidth', 0.9);
    grid on; xlabel('Maturity (months)'); ylabel('Yield (monthly decimals)');
    title('A/n vs. observable macro vs. latent');
    legend('Intercept (A/n)','Observable macro','Latent','Location','best');
end
