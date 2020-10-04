using SpectralMethodFFT


struct TimeIntegTools

    klm_A::KFunc
    klm_B::KFunc
    klm_C::KFunc

    function TimeIntegTools(klm_A, klm_B, klm_C)
        new(klm_A, klm_B, klm_C)
    end

    function TimeIntegTools(
        Re⁻¹::Real, dt::Real, c::ConfigFFT)

        xlens = xlens_xranges(c.xranges)

        klm_K = klm_Kgen(c)
        klm_L = klm_Lgen(c)
        klm_M = klm_Mgen(c)

        klm_A = exp(
            - Re⁻¹ * dt / 2.0
            * sum(
                (2π .* [klm_K, klm_L, klm_M] ./ xlens).^2
            )
        )

        klm_B = exp(
            - Re⁻¹ * dt / 2.0
            * (2π * klm_L / xlens[2])^2
        )

        function f(x)
            if abs(x) < eps()
                dt/2.0 - x*dt^2/8.0
            else
                (1 - exp(-x*dt/2.0)) / x
            end
        end

        xyz_Y = xyz_Ygen(c)
        klm_F = klm_xyz(sin(π/2 * xyz_Y))

        klm_C = KFunc(f.(Re⁻¹ * klm_L.vals), c) * klm_F

        TimeIntegTools(klm_A, klm_B, klm_C)
    end

end

function K_K_K_K_lin_K_K_K_K(
        klm_Ψ::KFunc{T,3},
        klm_Φ::KFunc{T,3},
        klm_Ux::KFunc{T,3},
        klm_Uz::KFunc{T,3},
        tit::TimeIntegTools
    ) where T

    return (
        tit.klm_A * klm_Ψ,
        tit.klm_A * klm_Φ,
        tit.klm_B * klm_Ux + tit.klm_C,
        tit.klm_B * klm_Uz
    )

end

function klm_Δxz_klm(f::KFunc{T,3}) where T

    return (
        (klm_∂x_klm ∘ klm_∂x_klm)(f)
        +
        (klm_∂z_klm ∘ klm_∂z_klm)(f)
    )

end

function klm_Δxz⁻¹_klm(f::KFunc{T,3}) where T

    c = f.config
    xlens = xlens_xranges(c.xranges)

    lap = - 4π^2 * (
        ( c.Kcoords[1] / xlens[1] ) .^ 2
        +
        ( c.Kcoords[3] / xlens[3] ) .^ 2
    )
    gvals = f.vals ./ lap

    zero_indiceses = [(
        map(c.ngrids) do x
            if x % 2 == 0
                return (1, x÷2 + 1)
            else
                return (1,)
            end
        end
    )...]
    zero_indiceses = [
        dim==2 ? tuple(1:c.ngrids[2]...) : zero_indiceses[dim]
        for dim = 1:3
    ]

    for indices in Iterators.product(zero_indiceses...)
        gvals[indices...] = 0.
    end

    return KFunc(gvals, c)

end

function K3_K3_uω_ΨΦUxUz_K_K_K_K(
        klm_Ψ::KFunc{T,3},
        klm_Φ::KFunc{T,3},
        klm_Ux::KFunc{T,3},
        klm_Uz::KFunc{T,3}
    ) where T

    klm_ψ = klm_Δxz⁻¹_klm(klm_Ψ)
    klm_uy = - klm_Δ⁻¹_klm(klm_Φ)
    klm_ϕ = - klm_Δxz⁻¹_klm(klm_uy)

    # --- u ---
    klm_ux = (
        - klm_∂z_klm(klm_ψ)
        + klm_∂x_klm(klm_∂y_klm(klm_ϕ))
        + klm_Ux
    )
    klm_uz = (
        + klm_∂x_klm(klm_ψ)
        + klm_∂z_klm(klm_∂y_klm(klm_ϕ))
        + klm_Uz
    )

    klm3_u = [
        klm_ux,
        klm_uy,
        klm_uz
    ]

    # --- ω ---
    klm3_ω = klm3_rot_klm3(klm3_u)

    return klm3_u, klm3_ω

end

function K_K_K_K_nonlin_K_K_K_K(
        klm_Ψ::KFunc{T,3},
        klm_Φ::KFunc{T,3},
        klm_Ux::KFunc{T,3},
        klm_Uz::KFunc{T,3}
    ) where T

    klm3_u, klm3_ω = (
        K3_K3_uω_ΨΦUxUz_K_K_K_K(
            klm_Ψ, klm_Φ, klm_Ux, klm_Uz
        )
    )

    klm_ux, klm_uy, klm_uz = klm3_u

    klm3_tmp1 = klm3_rot_klm3(klm3_ω × klm3_u)
    klm3_tmp2 = - klm3_rot_klm3(klm3_tmp1)

    klm_nonlin3 = - klm_∂y_klm(klm_mean_xz_klm(
        klm_uy ∗ klm_ux
    ))
    klm_nonlin4 = - klm_∂y_klm(klm_mean_xz_klm(
        klm_uy ∗ klm_uz
    ))

    return (
        klm3_tmp1[2],
        klm3_tmp2[2],
        klm_nonlin3,
        klm_nonlin4
    )

end

function klm_mean_xz_klm(f::KFunc{T,3}) where T

    ngrids = f.config.ngrids
    max_nwaves = (
        0, ngrids[2] ÷ 2 + 1, 0
    )
    return klm_lowpass_klm(f, max_nwaves)

end

function K_K_K_K_RK4_K_K_K_K(
        klm_Ψ::KFunc{T,3},
        klm_Φ::KFunc{T,3},
        klm_Ux::KFunc{T,3},
        klm_Uz::KFunc{T,3},
        tit::TimeIntegTools,
        dt::Real
    ) where T

    klm4_ΨΦUxUz = (
        klm_Ψ, klm_Φ, klm_Ux, klm_Uz
    )

    klm4_ΨΦUxUz1 = (
        dt .* K_K_K_K_nonlin_K_K_K_K(
            klm4_ΨΦUxUz...
        )
    )

    klm4_ΨΦUxUz2 = (
        dt .* K_K_K_K_nonlin_K_K_K_K(
            K_K_K_K_lin_K_K_K_K(
                (
                    klm4_ΨΦUxUz .+ klm4_ΨΦUxUz1 ./ 2
                )...,
                tit
            )...
        )
    )

    klm4_ΨΦUxUz3 = (
        dt .* K_K_K_K_nonlin_K_K_K_K(
            (
                K_K_K_K_lin_K_K_K_K(
                    klm4_ΨΦUxUz..., tit
                )
                .+ klm4_ΨΦUxUz2 ./ 2
            )...
        )
    )

    klm4_ΨΦUxUz4 = (
        dt .* K_K_K_K_nonlin_K_K_K_K(
            K_K_K_K_lin_K_K_K_K(
                (
                    K_K_K_K_lin_K_K_K_K(
                        klm4_ΨΦUxUz..., tit
                    )
                    .+ klm4_ΨΦUxUz3
                )...,
                tit
            )...
        )
    )


    return (
        K_K_K_K_lin_K_K_K_K(
            (
                K_K_K_K_lin_K_K_K_K(
                    (
                        klm4_ΨΦUxUz .+ klm4_ΨΦUxUz1 ./ 6
                    )...,
                    tit
                )
                .+
                (
                    klm4_ΨΦUxUz2 .+ klm4_ΨΦUxUz3
                ) ./ 3
            )...,
            tit
        )
        .+ klm4_ΨΦUxUz4 ./ 6
    )

end


function main()

    # *************************
    #   Setting
    # *************************
    # --- Space ---
    xranges = ((0., 16.), (-1., 3.), (0., 16.))
    ngrids = (64, 16, 64)

    # --- Time ---
    t_st = 0.0
    t_ed = 1.0
    nt = 1000
    dt = (t_ed - t_st) / nt
    Re⁻¹ = 1 / (2.56 * 180)

    # *************************
    #   Initialization
    # *************************
    config = ConfigFFT(ngrids, xranges)
    tit = TimeIntegTools(Re⁻¹, dt, config)

    xyz_X = xyz_Xgen(config)
    xyz_Y = xyz_Ygen(config)
    xyz_Z = xyz_Zgen(config)

    xyz_ψ = sin(π/2 * xyz_Y)*sin(exp(π*xyz_X))*cos(exp(π*xyz_Z))
    xyz_ϕ = sin(π/2 * xyz_Y)*cos(exp(π*xyz_X))*sin(exp(π*xyz_Z))
    xyz_Ux = XFunc(zeros(config.ngrids), config)
    xyz_Uz = XFunc(zeros(config.ngrids), config)

    klm_ψ = klm_xyz(xyz_ψ)
    klm_ϕ = klm_xyz(xyz_ϕ)
    klm_Ux = klm_xyz(xyz_Ux)
    klm_Uz = klm_xyz(xyz_Uz)

    klm_Ψ = klm_Δxz_klm(klm_ψ)
    klm_Φ = klm_Δ_klm(klm_Δxz_klm(klm_ϕ))

    for it = 0:nt

        # output here
        println("it=", it)
        klm3_u, klm3_ω = K3_K3_uω_ΨΦUxUz_K_K_K_K(
            klm_Ψ, klm_Φ, klm_Ux, klm_Uz
        )
        xyz3_u = xyz_klm.(klm3_u)
        println(xyz3_u[2][1, 3, 1])

        if it < nt
            klm_Ψ, klm_Φ, klm_Ux, klm_Uz = (
                K_K_K_K_RK4_K_K_K_K(
                    klm_Ψ, klm_Φ, klm_Ux, klm_Uz,
                    tit, dt
                )
            )
        end
    end

end

main()
