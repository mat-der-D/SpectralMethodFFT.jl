# *******************************************
#  Elementary Functions
# *******************************************
const ELEMFUNC = (
    :sin, :cos, :tan, :cot, :sec, :csc,
    :sinh, :cosh, :tanh, :coth, :sech, :csch,
    :asin, :acos, :atan, :acot, :asec, :acsc,
    :asinh, :acosh, :atanh, :acoth, :asech, :acsch,
    :sinpi, :cospi, :sinc, :cosc,
    :exp, :log, :sqrt, :cbrt,
    :abs
)

for fn = ELEMFUNC
    @eval begin
        Base.$fn(f::XFunc) = XFunc($fn.(f.vals), f.config)
        Base.$fn(f::KFunc) = KFunc($fn.(f.vals), f.config)
    end
end


# *******************************************
#  De-aliased Products
# *******************************************
# 3/2-rule ... zero-padding
# 2/3-rule ... truncating

# de-aliased product by 3/2-rule (zero padding)
function K_dealiasedprod_32_K_K(
            f::KFunc{T,N}, g::KFunc{T,N}
        ) where N where T

    if !(f.config === g.config)
        return println("ERROR")
    end

    fvals_pad = padding(f)
    gvals_pad = padding(g)

    P_fft = f.config.P_fftpad
    P_ifft = f.config.P_ifftpad
    if T <: Real
        fgvals_pad = P_fft * (
            real(P_ifft * fvals_pad) .* real(P_ifft * gvals_pad)
        )
    else
        fgvals_pad = P_fft * (
            (P_ifft * fvals_pad) .* (P_ifft * gvals_pad)
        )
    end
    fgvals = truncate(fgvals_pad, f.config)

    return KFunc(fgvals, f.config)

end

function padding(f::KFunc)
    config = f.config

    ngrids = config.ngrids
    pad_ngrids = @. ngrids ÷ 2 * 3

    slices = slices_padded_core(ngrids, pad_ngrids)

    vals_shift = fftshift(f.vals)
    padded = zeros(Complex{Float64}, pad_ngrids)
    padded[slices...] = vals_shift
    padded .= ifftshift(padded)

    return padded
end

function truncate(
            padded::Array{Complex{Float64},N},
            config::ConfigFFT{T,N}
        ) where N where T

    ngrids = config.ngrids
    pad_ngrids = @. ngrids ÷ 2 * 3

    slices = slices_padded_core(ngrids, pad_ngrids)

    nshifts = @. pad_ngrids÷2 - ngrids÷2
    min_nwaves = @. nshifts + 1
    max_nwaves = @. nshifts + ngrids
    slices = (:).(min_nwaves, max_nwaves)

    vals = ifftshift(fftshift(padded)[slices...])

    N_origin = prod(ngrids)
    N_padded = prod(pad_ngrids)
    vals *= N_padded / N_origin

    return vals

end

# helper function
function slices_padded_core(
        ngrids::NTuple{N1,Int},
        pad_ngrids::NTuple{N2,Int}
    ) where N1 where N2

    nshifts = @. pad_ngrids÷2 - ngrids÷2
    min_nwaves = @. nshifts + 1
    max_nwaves = @. nshifts + ngrids
    return (:).(min_nwaves, max_nwaves)

end

# de-aliased product by 2/3-rule (truncation)
function K_dealiasedprod_23_K_K(f::KFunc, g::KFunc)

    if !(f.config === g.config)
        return println("ERROR")
    end
    config = f.config
    ngrids = config.ngrids
    max_nwaves = @. ngrids ÷ 3
    f_trunc = K_lowpass_K(f, max_nwaves)
    g_trunc = K_lowpass_K(g, max_nwaves)
    return K_X(X_K(f_trunc) * X_K(g_trunc))

end

# +++++ aliases +++++
# \odot
⊙(f::KFunc, g::KFunc) = K_dealiasedprod_32_K_K(f, g)
# \otimes
⊗(f::KFunc, g::KFunc) = K_dealiasedprod_23_K_K(f, g)


# *******************************************
#  Analysis Tools
# *******************************************
function dVgen(config::ConfigFFT)
    xlens = (x -> -(.-(x...))).(config.xranges)
    return prod(xlens ./ config.ngrids)
end

function integ_X(f::XFunc)
    sum(f) * dVgen(f.config)
end

∫(f) = integ_X(f)

function norm_X(f::XFunc, p::Real=2)

    if p == Inf
        return max(abs(f)...)
    else
        return ( ∫( abs(f)^p ) )^(1/p)
    end

end

function l2inpr_X_X(f::XFunc{T,N}, g::XFunc{T,N}) where N where T

    if f.config === g.config
        return ∫(f * g)
    else
        println("ERROR")
    end

end
