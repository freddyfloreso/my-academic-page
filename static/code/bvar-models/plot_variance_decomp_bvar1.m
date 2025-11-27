function plot_variance_decomp_bvar1(VD)
    figure('Name','Variance decomposition across maturities');
    bar(VD.nList, [VD.share_macro, VD.share_latent, VD.share_meas], 'stacked','BarWidth',0.9);
    grid on; xlabel('Maturity (months)'); ylabel('Share of variance');
    title('Unconditional variance shares: macro | latent | meas');
    legend('Macro','Latent','Measurement','Location','best');
end
