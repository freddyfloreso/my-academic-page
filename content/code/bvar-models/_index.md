+++
date = '2025-11-23T06:12:12-05:00'
draft = false
title = 'BVAR Models'
+++

Below are my Bayesian VAR curve estimation for Peruvian yield curve modelling modules, uploaded for replication:

Main code:

- [BVAR_Macro_Est_vf.mlx](/code/bvar-models/BVAR_Macro_Est_vf.mlx)


The function support codes:

- [ap_two_step_bvar.m](/code/bvar-models/ap_two_step_bvar.m)
- [ap_two_step_bvar_p8.m](/code/bvar-models/ap_two_step_bvar_p8.m)
- [ap_two_step_var.m](/code/bvar-models/ap_two_step_var.m)
- [ap_two_step_var_p8.m](/code/bvar-models/ap_two_step_var_p8.m)
- [bvar1_minnesota.m](/code/bvar-models/bvar1_minnesota.m)
- [bvarp_minnesota_p8.m](/code/bvar-models/bvarp_minnesota_p8.m)
- [get_ATSM_loadings_bvar1.m](/code/bvar-models/get_ATSM_loadings_bvar1.m)
- [make_loading_table_bvar1.m](/code/bvar-models/make_loading_table_bvar1.m)
- [plot_contrib_cross_section_bvar1.m](/code/bvar-models/plot_contrib_cross_section_bvar1.m)
- [plot_contrib_timeseries_bvar1.m](/code/bvar-models/plot_contrib_timeseries_bvar1.m)
- [plot_loading_curves_bvar1.m](/code/bvar-models/plot_loading_curves_bvar1.m)
- [plot_variance_decomp_bvar1.m](/code/bvar-models/plot_variance_decomp_bvar1.m)
- [robust_z.m](/code/bvar-models/robust_z.m)
- [smooth_latent.m](/code/bvar-models/smooth_latent.m)
- [summarize.m](/code/bvar-models/summarize.m)
- [var_irf_chol.m](/code/bvar-models/var_irf_chol.m)
- [variance_decomp_bvar1.m](/code/bvar-models/variance_decomp_bvar1.m)
- [yield_contrib_at_t_bvar1.m](/code/bvar-models/yield_contrib_at_t_bvar1.m)

The input files:

- [Des_Macro_Vars.xlsx](/code/bvar-models/Des_Macro_Vars.xlsx)
- [Fuente_Input_Empleo_Mensuales-20250820-192845.xlsx](/code/bvar-models/Fuente_Input_Empleo_Mensuales-20250820-192845.xlsx)
- [Fuente_Input_IPC_Alimentos_En_Mensuales-20250820-221429.xlsx](/code/bvar-models/Fuente_Input_IPC_Alimentos_En_Mensuales-20250820-221429.xlsx)
- [Fuente_Input_IPC_Importado_Mensuales-20250820-221503.xlsx](/code/bvar-models/Fuente_Input_IPC_Importado_Mensuales-20250820-221503.xlsx)
- [Fuente_Input_IPC_Mensuales-20250820-215828.xlsx](/code/bvar-models/Fuente_Input_IPC_Mensuales-20250820-215828.xlsx)
- [Fuente_Input_PBI_Ind_No_Primario_Mensuales-20250820-210626.xlsx](/code/bvar-models/Fuente_Input_PBI_Ind_No_Primario_Mensuales-20250820-210626.xlsx)
- [Fuente_Input_PBI_Ind_Primario_Mensuales-20250820-211754.xlsx](/code/bvar-models/Fuente_Input_PBI_Ind_Primario_Mensuales-20250820-211754.xlsx)
- [Fuente_Input_curva_historica.xlsx](/code/bvar-models/Fuente_Input_curva_historica.xlsx)
- [Fuente_Input_curva_historica2.xlsx](/code/bvar-models/Fuente_Input_curva_historica2.xlsx)
- [Fuente_USA_GS10.csv](/code/bvar-models/Fuente_USA_GS10.csv)
- [GS_Consol_Int.xlsx](/code/bvar-models/GS_Consol_Int.xlsx)
- [Macro_Data_Consolidated.xlsx](/code/bvar-models/Macro_Data_Consolidated.xlsx)
- [Vector_Precios.xlsm](/code/bvar-models/Vector_Precios.xlsm)
