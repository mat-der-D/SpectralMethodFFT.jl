module SpectralMethodFFT

include("generalFFT.jl")

export ConfigFFT, XFunc, KFunc

export x_Xgen, xy_Xgen, xy_Ygen, xyz_Xgen, xyz_Ygen, xyz_Zgen,
       k_Kgen, kl_Kgen, kl_Lgen, klm_Kgen, klm_Lgen, klm_Lgen

export pass_K!, highpass_K!, K_highpass_K, lopass_K!, K_lowpass_K,
       k_highpass_k, kl_highpass_kl, klm_highpass_klm,
       k_lowpass_k, kl_lowpass_kl, klm_lowpass_klm

export K_dealiasedprod_32_K_K, K_dealiasedprod_23_K_K, ⊙, ⊗

export K_X, k_x, kl_xy, klm_xyz, X_K, x_k, xy_kl, xyz_klm

export K_∂Xaxis_K, k_∂x_k, kl_∂x_kl, kl_∂y_kl,
       klm_∂x_klm, klm_∂y_klm, klm_∂z_klm,
       X_∂Xaxis_X, x_∂x_x, xy_∂x_xy, xy_∂y_xy,
       xyz_∂x_xyz, xyz_∂y_xyz, xyz_∂z_xyz

export K_laplacian_K, K_Δ_K,
       k_laplacian_k, kl_laplacian_kl, klm_laplacian_klm,
       k_Δ_k, kl_Δ_kl, klm_Δ_klm,
       X_laplacian_X, X_Δ_X,
       x_laplacian_x, xy_laplacian_xy, xyz_laplacian_xyz,
       x_Δ_x, xy_Δ_xy, xyz_Δ_xyz

export kl2_grad_kl, kl_rot_kl2, kl_div_kl2,
       klm3_grad_klm, klm3_rot_klm3, klm_div_klm3

export integ_X, integ_x, integ_xy, integ_xyz,
       norm_X, norm_x, norm_xy, norm_xyz,
       l2inpr_X_X, l2inpr_x_x, l2inpr_xy_xy, l2inpr_xyz_xyz

end
