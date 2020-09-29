# *******************************************
#  Coordinate Tools for specific dimensions
# *******************************************
# +++++ Coordinate in X-space ++++++++++
const XKGENS = (
    (:x_Xgen, :k_Kgen, 1, 1),
    (:xy_Xgen, :kl_Kgen, 2, 1),
    (:xy_Ygen, :kl_Lgen, 2, 2),
    (:xyz_Xgen, :klm_Kgen, 3, 1),
    (:xyz_Ygen, :klm_Lgen, 3, 2),
    (:xyz_Zgen, :klm_Mgen, 3, 3)
)

for (fnx, fnk, dim, axis) = XKGENS
    @eval begin
        # X generators
        function $fnx(config::ConfigFFT{T,$dim}) where T
            XFunc(copy(config.Xcoords[$axis]), config)
        end

        # K generators
        function $fnk(config::ConfigFFT{T,$dim}) where T
            KFunc(copy(config.Kcoords[$axis]), config)
        end
    end
end


# *******************************************
#  Low/High-pass Filters
# *******************************************
const PASSFILTERS = (
    (:k_highpass_k, 1, :K_highpass_K),
    (:kl_highpass_k, 2, :K_highpass_K),
    (:klm_highpass_klm, 3, :K_highpass_K),
    (:k_lowpass_k, 1, :K_lowpass_K),
    (:kl_lowpass_kl, 2, :K_lowpass_K),
    (:klm_lowpass_klm, 3, :K_lowpass_K)
)

for (fn, dim, fnK) = PASSFILTERS
    @eval begin
        KF = KFunc{T,$dim} where T
        NT = NTuple{$dim,Int}
        $fn(f::KF, nwaves::NT) = $fnK(f, nwaves)
    end
end


# *******************************************
#  Fourier Transformation
# *******************************************
const FFTS = (
    (:k_x, :x_k, 1),
    (:kl_xy, :xy_kl, 2),
    (:klm_xyz, :xyz_klm, 3)
)
for (kx, xk, dim) = FFTS
    @eval $kx(f::XFunc{T,$dim} where T) = K_X(f)
    @eval $xk(f::KFunc{T,$dim} where T) = X_K(f)
end


# *******************************************
#  Diffrential Operators
# *******************************************
# +++++ basics +++++
const DIFFS = (
    (:k_∂x_k, :x_∂x_x, 1, 1),
    (:kl_∂x_kl, :xy_∂x_xy, 2, 1),
    (:kl_∂y_kl, :xy_∂y_xy, 2, 2),
    (:klm_∂x_klm, :xyz_∂x_xyz, 3, 1),
    (:klm_∂y_klm, :xyz_∂y_xyz, 3, 2),
    (:klm_∂z_klm, :xyz_∂z_xyz, 3, 3)
)

for (opk, opx, dim, axis) = DIFFS
    @eval begin
        KF = KFunc{T,$dim} where T
        $opk(f::KF) = K_∂Xaxis_K(f, $axis)
        XF = XFunc{T,$dim} where T
        $opx(f::XF) = X_∂Xaxis_X(f, $axis)
    end
end

# +++++ laplacian +++++
const LAPLA = (
    (:k_laplacian_k, :x_laplacian_x,
        :k_Δ_k, :x_Δ_x, 1),
    (:kl_laplacian_kl, :xy_laplacian_xy,
        :kl_Δ_kl, :xy_Δ_xy, 2),
    (:klm_laplacian_klm, :xyz_laplacian_xyz,
        :klm_Δ_klm, :xyz_Δ_xyz, 3),
)

for (klap, xlap, kΔ, xΔ, dim) = LAPLA
    @eval begin
        KF = KFunc{T,$dim} where T
        $klap(f::KF) = K_laplacian_K(f)
        XF = XFunc{T,$dim} where T
        $xlap(f::XF) = X_laplacian_X(f)
        $kΔ = $klap
        $xΔ = $xlap
    end
end

# +++++ laplacian +++++
const LAPINV = (
    (:k_laplainv_k, :x_laplainv_x,
        :k_Δ⁻¹_k, :x_Δ⁻¹_x, 1),
    (:kl_laplainv_kl, :xy_laplainv_xy,
        :kl_Δ⁻¹_kl, :xy_Δ⁻¹_xy, 2),
    (:klm_laplainv_klm, :xyz_laplainv_xyz,
        :klm_Δ⁻¹_klm, :xyz_Δ⁻¹_xyz, 3),
)

for (klinv, xlinv, kΔ⁻¹, xΔ⁻¹, dim) = LAPINV
    @eval begin
        KF = KFunc{T,$dim} where T
        $klinv(f::KF) = K_laplainv_K(f)
        XF = XFunc{T,$dim} where T
        $xlinv(f::XF) = X_laplainv_X(f)
        $kΔ⁻¹ = $klinv
        $xΔ⁻¹ = $xlinv
    end
end


# +++++ vector analysis +++++
# 2-dimensional
function kl2_grad_kl(kl_func::KFunc{T,2})::Vector{KFunc{T,2}} where T
    return [
        kl_∂x_kl(kl_func)
        kl_∂y_kl(kl_func)
    ]
end

function kl_rot_kl2(
            kl2_func::Vector{KFunc{T,2}}
        )::KFunc{T,2} where T

    if length(kl2_func) != 2
        return println("ERROR")
    end
    return (
        kl_∂x_kl(kl2_func[2])
        - kl_∂y_kl(kl2_func[1])
    )
end

function kl_div_kl2(
            kl2_func::Vector{KFunc{T,2}}
        )::KFunc{T,2} where T

    if length(kl2_func) != 2
        return println("ERROR")
    end
    return (
        kl_∂x_kl(kl2_func[1])
        + kl_∂y_kl(kl2_func[2])
    )
end

# 3-dimensional
function klm3_grad_klm(
            klm_func::KFunc{T,3}
        )::Vector{KFunc{T,3}} where T

    return [
        klm_∂x_klm(klm_func)
        klm_∂y_klm(klm_func)
        klm_∂z_klm(klm_func)
    ]

end

function klm3_rot_klm(
            klm3_func::Vector{KFunc{T,3}}
        )::Vector{KFunc{T,3}} where T

    if length(klm3_func) != 3
        return println("ERROR")
    end
    return [
        klm_∂y_klm(klm3_func[3]) - klm_∂z_klm(klm3_func[2])
        klm_∂z_klm(klm3_func[1]) - klm_∂x_klm(klm3_func[3])
        klm_∂x_klm(klm3_func[2]) - klm_∂y_klm(klm3_func[1])
    ]

end

function klm_div_klm3(
            klm3_func::Vector{KFunc{T,3}}
        )::KFunc{T,3} where T

    if length(klm3_func) != 3
        return println("ERROR")
    end
    return (
        klm_∂x_klm(klm3_func[1])
        + klm_∂y_klm(klm3_func[2])
        + klm_∂z_klm(klm3_func[3])
    )

end


# *******************************************
#  Analysis Tools
# *******************************************
const TOOLS = (
    (:integ_x, :norm_x, :l2inpr_x_x, 1),
    (:integ_xy, :norm_xy, :l2inpr_xy_xy, 2),
    (:integ_xyz, :norm_xyz, :l2inpr_xyz_xyz, 3)
)

for (integ, norm, inpr, dim) = TOOLS
    @eval begin
        XF = XFunc{T,$dim} where T
        $integ(f::XF) = integ_X(f)
        $norm(f::XF, p::Real=2) = norm_X(f, p)
        $inpr(f::XF, g::XF) = l2inpr_X_X(f, g)
    end
end
