using SpectralMethodFFT
using Test

@testset "SpectralMethodFFT.jl" begin

    # +++++ 1-dimensional +++++
    c = ConfigFFT((64,), ((0., 2π),))
    x_X = x_Xgen(c)
    x_f, x_g = exp(cos(x_X)), exp(sin(x_X))
    x_fg_exact = exp(cos(x_X) + sin(x_X))

    k_f, k_g = k_x.((x_f, x_g))
    x_fg = x_k(k_f ⊙ k_g)
    @test norm_x(x_fg - x_fg_exact) / (2π) < 2eps()
    x_fg = x_k(k_f ⊗ k_g)
    @test norm_x(x_fg - x_fg_exact) / (2π) < 2eps()

    # +++++ 2-dimensional +++++
    c = ConfigFFT((64, 64), ((0., 2π), (0., 2π)))
    xy_X, xy_Y = xy_Xgen(c), xy_Ygen(c)
    xy_f, xy_g = exp(cos(xy_X)), exp(cos(xy_Y))
    xy_fg_exact = eps(cos(xy_X) + cos(xy_Y))

    kl_f, kl_g = kl_xy.((xy_f, xy_g))
    xy_fg = xy_kl(kl_f ⊙ kl_g)
    @test norm_xy(xy_fg - xy_fg_exact) / ((2π)^2) < eps()
    xy_fg = xy_kl(kl_f ⊗ kl_g)
    @test norm_xy(xy_fg - xy_fg_exact) / ((2π)^2) < eps()

    # +++++ 3-dimensional +++++
    c = ConfigFFT((64, 64, 64), ((0., 2π), (0., 2π), (0., 2π)))
    xyz_X, xyz_Y, xyz_Z = xyz_Xgen(c), xyz_Ygen(c), xyz_Zgen(c)
    xyz_f = exp(cos(xyz_X) + cos(xyz_Y))
    xyz_g = exp(cos(xyz_Y) + cos(xyz_Z))
    xyz_fg_exact = exp(cos(xyz_X) + 2cos(xyz_Y) + cos(xyz_Z))

    klm_f, klm_g = klm_xyz.((xyz_f, xyz_g))
    xyz_fg = xyz_klm(klm_f ⊙ klm_g)
    @test norm_xyz(xyz_fg - xyz_fg_exact) / ((2π)^3) < eps()
    xyz_fg = xyz_klm(klm_f ⊗ klm_g)
    @test norm_xyz(xyz_fg - xyz_fg_exact) / ((2π)^3) < eps()

end
