# fft version
using FFTW


# *******************************************
#  Configuration
# *******************************************
"""
    ConfigFFT{T <: Union{Float64,Complex{Float64}},N}

Assembly all basic configurations of the Fourier-spectral method.
T represents the type of values of functions in 'real' space,
and N represents the dimension. The followings are all members:

    ngrids::NTuple{N,Int}
... Each element means the number of grids in each axis

    xranges::NTuple{N,NTuple{2,Float64}}
... Tuple of ((xmin, xmax), (ymin, ymax), ...)

    Xcoords::NTuple{N,Array{Float64,N}}
... Coordinates of 'real' space in each axis

    Kcoords::NTuple{N,Array{Complex{Float64},N}}
... Wavenumbers in each axis. It is used to calculate derivatives.

    P_fft, P_ifft, P_fftpad, P_ifftpad
... Operators used to accelerate fft and ifft

    cut_zigzag_mode::Bool
... The highest mode is suppressed or not for axes whose grids number is even.
If you use the easy constructor (explained below), it is set to be 'true' by default.
This flag is used in 'X_K' and 'K_X' function, the fourier transformation. If you
try to accelerate as much as possible and can manage the highest wavenumber mode,
set 'cut_zigzag_mode=false'.

For normal use, the easy constructor below is recommended for initialization.

    ConfigFFT(
        ngrids::NTuple{N,Int},
        xranges::NTuple{N,NTuple{2,Float64}};
        use_complex::Bool=false,
        cut_zigzag_mode::Bool=true)

The function in 'real' space is set to be real-valued (Float64) by default.
If you want to use complex-valued function, set 'use_complex=true'.

# Examples
1) 1-dimensional case. The number of grid is 64, and the range is 0 ≦ x < 10.

    julia> ngrids = (64,);

    julia> xranges = ((0., 10.),);

    julia> config = ConfigFFT(ngrids, xranges);

2) 2-dimensional case. The numbers of grids are 128 in x-axis and 256 in y-axis.
   The ranges in each axis are 0 ≦ x < 2π and 0 ≦ y < 4π.

    julia> ngrids = (128, 256);

    julia> xranges = ((0., 2π), (0., 4π));

    julia> config = ConfigFFT(ngrids, xranges);

3) 1-dimensional case. The number of grid is 64, and the range is 0 ≦ x < 30.
   The complex-valued functions in 'real' space is used.

   julia> ngrids = (64,);

   julia> xranges = ((0., 30.),);

   julia> config = ConfigFFT(ngrids, xranges, use_complex=true);
"""
struct ConfigFFT{T<:Union{Float64,Complex{Float64}},N}

    ngrids::NTuple{N,Int} # Number Of Grids

    xranges::NTuple{N,NTuple{2,Float64}} # tuple of (min, max) s
    Xcoords::NTuple{N,Array{Float64,N}}
    Kcoords::NTuple{N,Array{Complex{Float64},N}}

    P_fft::FFTW.cFFTWPlan{T1} where T1
    P_ifft::AbstractFFTs.ScaledPlan{T2} where T2
    P_fftpad::FFTW.cFFTWPlan{T3} where T3
    P_ifftpad::AbstractFFTs.ScaledPlan{T4} where T4

    cut_zigzag_mode::Bool

    # CONSTRUCTOR
    function ConfigFFT{T,N}(
            ngrids, xranges, Xcoords, Kcoords,
            P_fft, P_ifft, P_fftpad, P_ifftpad;
            cut_zigzag_mode=true
        ) where {T,N}

        warn_if_ngrid_is_not_power_of_2(ngrids)

        new(ngrids, xranges, Xcoords, Kcoords,
            P_fft, P_ifft, P_fftpad, P_ifftpad,
            cut_zigzag_mode)
    end

    # EASY CONSTRUCTOR
    function ConfigFFT(
            ngrids::NTuple{N,Int},
            xranges::NTuple{N,NTuple{2,Float64}};
            use_complex::Bool=false,
            cut_zigzag_mode::Bool=true
        ) where N

        T = use_complex ? Complex{Float64} : Float64

        carts = CartesianIndices(ngrids)
        Xcoords = Xcoordsgen(ngrids, xranges)
        Kcoords = Kcoordsgen(ngrids, xranges)

        P_fft = plan_fft(Xcoords[1])
        P_ifft = plan_ifft(Xcoords[1])

        pad_ngrids = @. ngrids ÷ 2 * 3
        P_fftpad = plan_fft(zeros(pad_ngrids))
        P_ifftpad = plan_ifft(zeros(pad_ngrids))

        return ConfigFFT{T,N}(
            ngrids, xranges, Xcoords, Kcoords,
            P_fft, P_ifft, P_fftpad, P_ifftpad,
            cut_zigzag_mode=cut_zigzag_mode)
    end

end

# --- helper function for constuctor ---
function Xcoordsgen(
        ngrids::NTuple{N,Int},
        xranges::NTuple{N,NTuple{2,Float64}}
    ) where N

    _Xcoordgen(axis) = Xcoordgen(axis, ngrids, xranges)
    return ntuple(_Xcoordgen, N)

end

function Xcoordgen(
        axis::Int,
        ngrids::NTuple{N,Int},
        xranges::NTuple{N,NTuple{2,Float64}}
    ) where N

    ngrid = ngrids[axis]
    xrange = xranges[axis]
    _Xcoordgen(indices) = (
        (  (indices[axis] - 1)*xrange[2]
         + (ngrid - indices[axis] + 1)*xrange[1] ) / ngrid
    )
    return _Xcoordgen.(CartesianIndices(ngrids))

end

function Kcoordsgen(
        ngrids::NTuple{N,Int},
        xranges::NTuple{N,NTuple{2,Float64}}
    ) where N

    _Kcoordgen(axis) = Kcoordgen(axis, ngrids, xranges)
    return ntuple(_Kcoordgen, N)

end

function Kcoordgen(
        axis::Int,
        ngrids::NTuple{N,Int},
        xranges::NTuple{N,NTuple{2,Float64}}
    ) where N

    ngrid = ngrids[axis]
    xrange = xranges[axis]
    _Kcoordgen(indices) = (
        kval(indices[axis], ngrid, xrange)
    )

    return _Kcoordgen.(CartesianIndices(ngrids))

end

function kval(index, ngrid, xrange)::Complex{Float64}

    index0 = index - 1  # start by 0
    if 2*index0 < ngrid
        return index0
    elseif 2*index0 == ngrid
        return 0
    elseif 2*index0 > ngrid
        return index0 - ngrid
    end

end


# *******************************************
#  XFunc
# *******************************************
"""
    XFunc{T,N} <: AbstractArray{T,N}

A struct of a function on 'real' space.
It consists of the following:

    vals::Array{T,N}
... Values of the function on grids

    config::ConfigFFT{T,N}
... Configuration of Fourier Transformation

For normal use, the easy constructor below is recommended for initialization.

    XFunc(
        vals::Array{Tc,N},
        config::ConfigFFT{T,N}
        ) where {N, Tc<:Number, T}

The type of elements of input 'vals' is converted to T.
Instead, you may generate a XFunc object, whose values are undef as

    XFunc(undef, config)

XFunc type object is transformed by 'K_X' to KFunc type object,
which represents a function in 'wavenumber' space.
See also the explanation of KFunc.

# Examples
Suppose 'config' be ConfigFFT{Float64,2} type object,
and c.ngrids = (2, 2). Then, for example,

    v = [1. 2.; 3. 4.]
    X_func = XFunc(v, config)

generates a XFunc object whose 'vals' is v.
If 'K_X' is applied to this function, say,

    K_func = K_X(X_func)

a new object 'K_func' is returned.
It is KFunc type, and its 'vals' is Fourier transform of v.
"""
mutable struct XFunc{T,N} <: AbstractArray{T,N}

    vals::Array{T,N}
    config::ConfigFFT{T,N}

    # CONSTRUCTOR
    function XFunc{T,N}(
            vals::Array{T,N},
            config::ConfigFFT{T,N}
        ) where {T,N}

        check_size_consistency(vals, config)
        return new(vals, config)

    end

    # EASY CONSTRUCTOR
    function XFunc(
            vals::Array{Tv,N},
            config::ConfigFFT{Tc,N}
        ) where {N, Tv<:Number, Tc}

        if Tv <: Complex && Tc <: Real
            return XFunc{Tc,N}(Tc.(real(vals)), config)
        else
            return XFunc{Tc,N}(Tc.(vals), config)
        end

    end

    # UNDEF CONSTRUCTOR
    function XFunc(
            undef::UndefInitializer,
            config::ConfigFFT{T,N}
        ) where {T,N}

        f_undef = Array{T,N}(undef, config.ngrids)
        return XFunc{T,N}(f_undef, config)

    end

end

Base.:size(f::XFunc) = size(f.vals)

Base.:getindex(f::XFunc, i::Int) = getindex(f.vals, i)

function Base.:getindex(
        f::XFunc{T,N}, I::Vararg{Int,N}
    ) where {T,N}

    getindex(f.vals, I...)
end

Base.:setindex!(f::XFunc, v, i::Int) = setindex!(f.vals, v, i)

function Base.:setindex!(
        f::XFunc{T,N}, v, I::Vararg{Int,N}
    ) where {T,N}

    setindex!(f.vals, v, I...)
end

Base.:copy(f::XFunc) = XFunc(copy(f.vals), f.config)


# *******************************************
#  KFunc
# *******************************************
"""
    KFunc{T,N} <: AbstractArray{T,N}

A struct of a function on 'wavenumber' space.
It consists of the following:

    vals::Array{Complex{Float64},N}
... Values of the function on grids

    config::ConfigFFT{T,N}
... Configuration of Fourier Transformation

For normal use, the easy constructor below is recommended for initialization.

    KFunc(
        vals::Array{Tc,N},
        config::ConfigFFT{T,N}
    ) where {N, Tc<:Number, T}

The type of elements of input 'vals' is converted to T.
Instead, you may generate a KFunc object, whose values are undef as

    KFunc(undef, config)

KFunc type object is transformed by 'X_K' to XFunc type object,
which represents a function in 'real' space.
See also the explanation of XFunc.

# Examples
Suppose 'config' be ConfigFFT{Float64,2} type object,
and c.ngrids = (2, 2). Then, for example,

    v = [1. 2.; 3. 4.]
    K_func = KFunc(v, config)

generates a XFunc object whose 'vals' is v.
If 'X_K' is applied to this function, say,

    X_func = X_K(K_func)

a new object 'X_func' is returned.
It is XFunc type, and its 'vals' is inverse-Fourier transform of v.
"""
mutable struct KFunc{T,N} <: AbstractArray{Complex{Float64},N}

    vals::Array{Complex{Float64},N}
    config::ConfigFFT{T,N}

    # CONSTRUCTOR
    function KFunc{T,N}(
            vals::Array{Complex{Float64},N},
            config::ConfigFFT{T,N}
        ) where {T,N}

        check_size_consistency(vals, config)
        return new(vals, config)

    end

    # EASY CONSTRUCTOR
    function KFunc(
            vals::Array{Tv,N},
            config::ConfigFFT{Tc,N}
        ) where {N, Tv<:Number, Tc}

        KFunc{Tc,N}(Complex{Float64}.(vals), config)
    end

    # UNDEF CONSTRUCTOR
    function KFunc(
            undef::UndefInitializer,
            config::ConfigFFT{T,N}
        ) where {T,N}

        f_undef = Array{Complex{Float64},N}(undef, config.ngrids)
        return KFunc{T,N}(f_undef, config)

    end

end

Base.:size(f::KFunc) = size(f.vals)

Base.:getindex(f::KFunc, i::Int) = getindex(f.vals, i)

function Base.:getindex(
        f::KFunc{T,N}, I::Vararg{Int,N}
    ) where {T,N}

    getindex(f.vals, I...)
end

Base.:setindex!(f::KFunc, v, i::Int) = setindex!(f.vals, v, i)

function Base.:setindex!(
        f::KFunc{T,N}, v, I::Vararg{Int,N}
    ) where {T,N}

    setindex!(f.vals, v, I...)
end

Base.:copy(f::KFunc) = KFunc(copy(f.vals), f.config)


# *******************************************
#  Binomial Operators
# *******************************************
const BINOP = (
    (:+, :.+), (:-, :.-), (:*, :.*),
    (:/, :./), (:\, :.\), (:^, :.^)
)

# +++++ XFunc +++++
Base.:+(f::XFunc) = f
Base.:-(f::XFunc) = XFunc(-f.vals, f.config)
Base.:^(f::XFunc, n::Integer) = XFunc(f.vals .^ n, f.config)
Base.:inv(f::XFunc) = f \ 1.0

for (op, opd) = BINOP
    @eval begin
        function Base.$op(f::XFunc, g::XFunc)
            check_config_consistency(f.config, g.config)
            return XFunc($opd(f.vals, g.vals), f.config)
        end

        Base.$op(f::XFunc, a::Number) = (
            XFunc($opd(f.vals, a), f.config)
        )
        Base.$op(a::Number, f::XFunc) = (
               XFunc($opd(a, f.vals), f.config)
        )
    end
end


# +++++ KFunc +++++
Base.:+(f::KFunc) = f
Base.:-(f::KFunc) = KFunc(-f.vals, f.config)
Base.:^(f::KFunc, n::Integer) = KFunc(f.vals .^ n, f.config)
Base.:inv(f::KFunc) = f \ 1.0

for (op, opd) = BINOP
    @eval begin
        function Base.$op(f::KFunc, g::KFunc)
            check_config_consistency(f.config, g.config)
            return KFunc($opd(f.vals, g.vals), f.config)
        end

        Base.$op(f::KFunc, a::Number) = (
            KFunc($opd(f.vals, a), f.config)
        )
        Base.$op(a::Number, f::KFunc) = (
            KFunc($opd(a, f.vals), f.config)
        )
    end
end


# *******************************************
#  Operators for Complex Numbers
# *******************************************
const OPERATOR = (
    :real, :imag, :reim, :conj
)

for op = OPERATOR
    @eval begin
        Base.$op(f::XFunc) = XFunc($op(f.vals), f.config)
        Base.$op(f::KFunc) = KFunc($op(f.vals), f.config)
    end
end


# *******************************************
#  Low/High-pass Filter
# *******************************************
# +++++ general pass filter +++++
function pass_K!(
        f::KFunc{T,N},
        slices::NTuple{N,UnitRange{Int}}
    ) where {T,N}

    vals = copy(f.vals)
    vals[slices...] .= 0.0 + 0.0im
    f.vals -= vals

end

# +++++ high-pass filter +++++
function highpass_K!(
        f::KFunc{T,N},
        min_nwaves::NTuple{N,Int}
    ) where {T,N}

    ngrids = f.config.ngrids
    if any(@. min_nwaves > ngrids ÷ 2)
        println("WARNING: all waves are suppressed")
        f.vals .= 0.
    else
        floors = tuple(ones(Int, N)...)
        ceils = ngrids
        min_indices = @. max(floors, floors + min_nwaves)
        max_indices = @. min(ceils, ceils + 1 - min_nwaves)

        slices = (:).(min_indices, max_indices)
        pass_K!(f, slices)
    end
end

function K_highpass_K(
        f::KFunc{T,N},
        min_nwaves::NTuple{N,Int}
    ) where {T,N}

    g = copy(f)
    highpass_K!(g, min_nwaves)
    return g

end

# +++++ low-pass filter +++++
function lowpass_K!(
        f::KFunc{T,N},
        max_nwaves::NTuple{N,Int}
    ) where {T,N}

    ngrids = f.config.ngrids
    f.vals .= fftshift(f.vals)

    center_indices = @. ngrids ÷ 2 + 1
    if any(max_nwaves .< 0)
        println("WARNING: all waves are suppressed")
        f.vals .= 0.
    else
        floors = tuple(ones(Int, N)...)
        ceils = ngrids
        min_indices = (
            @. max(floors, center_indices - max_nwaves)
        )
        max_indices = (
            @. min(ceils, center_indices + max_nwaves)
        )

        slices = (:).(min_indices, max_indices)
        pass_K!(f, slices)
    end

    f.vals .= ifftshift(f.vals)

end

function K_lowpass_K(
        f::KFunc{T,N},
        max_nwaves::NTuple{N,Int}
    ) where {T,N}

    g = copy(f)
    lowpass_K!(g, max_nwaves)
    return g

end


# *******************************************
#  Fourier Transformation
# *******************************************
function K_X(f::XFunc{T,N}) where {T,N}

    P = f.config.P_fft
    g = KFunc(P * f.vals, f.config)

    if f.config.cut_zigzag_mode
        ngrids = f.config.ngrids
        max_nwaves = @. ngrids ÷ 2 - 1
        lowpass_K!(g, max_nwaves)
    end

    return g
end

function X_K(f::KFunc{T,N}) where {T,N}

    P = f.config.P_ifft
    if T <: Real
        XFunc(real(P * f.vals), f.config)
    else
        XFunc(P * f.vals, f.config)
    end
end


# *******************************************
#  Differentiation
# *******************************************
# +++++ destructive +++++
function ∂Xaxis_K!(f::KFunc, axis::Int)

    config = f.config
    xrange = config.xranges[axis]
    xlen = xrange[2] - xrange[1]
    Kcoord = config.Kcoords[axis]

    f.vals .*= (2π*im/xlen)*Kcoord

end

# +++++ non-destructive +++++
function K_∂Xaxis_K(f::KFunc, axis::Int)
    g = copy(f)
    ∂Xaxis_K!(g, axis)
    return g
end

function X_∂Xaxis_X(f::XFunc, axis::Int)
    X_K(K_∂Xaxis_K(K_X(f), axis))
end

function K_laplacian_K(f::KFunc{T,N}) where {T,N}

    return sum(
        K_∂Xaxis_K(K_∂Xaxis_K(f, axis), axis)
        for axis = 1:N
    )

end

K_Δ_K = K_laplacian_K
X_laplacian_X = X_K ∘ K_laplacian_K ∘ K_X
X_Δ_X = X_laplacian_X

"""
K_laplainv_K returns g satisfying Δg = f
"""
function K_laplainv_K(f::KFunc{T,N}) where {T,N}

    c = f.config
    xlens = xlens_xranges(c.xranges)

    lap = - 4π^2 * sum(
        (x -> x .^ 2).(c.Kcoords) ./ (xlens .^ 2)
    )
    lap_inv = 1 ./ lap
    gvals = lap_inv .* f.vals

    zero_coords = (
        map(c.ngrids) do x
            if x % 2 == 0
                return (1, x÷2 + 1)
            else
                return (1,)
            end
        end
    )
    for icoords in Iterators.product(zero_coords...)
        gvals[icoords...] = 0.
    end

    return KFunc(gvals, c)

end

K_Δ⁻¹_K = K_laplainv_K
X_laplainv_X = X_K ∘ K_laplainv_K ∘ K_X
X_Δ⁻¹_X = X_laplainv_X


# *******************************************
#  De-aliased Products
# *******************************************
# 3/2-rule ... zero-padding
# 2/3-rule ... truncating

# de-aliased product by 3/2-rule (zero padding)
function K_dealiasedprod_32_K_K(
        f::KFunc{T,N}, g::KFunc{T,N}
    ) where {T,N}

    check_config_consistency(f.config, g.config)

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
    pad_ngrids = to_pad_ngrids(ngrids)

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
    ) where {T,N}

    ngrids = config.ngrids
    pad_ngrids = to_pad_ngrids(ngrids)

    slices = slices_padded_core(ngrids, pad_ngrids)
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
    ) where {N1,N2}

    nshifts = @. pad_ngrids÷2 - ngrids÷2
    min_nwaves = @. nshifts + 1
    max_nwaves = @. nshifts + ngrids
    return (:).(min_nwaves, max_nwaves)

end

function to_pad_ngrids(ngrids::NTuple{N,Int}) where N
    @. ngrids ÷ 2 * 3
end


# de-aliased product by 2/3-rule (truncation)
function K_dealiasedprod_23_K_K(f::KFunc, g::KFunc)

    check_config_consistency(f.config, g.config)
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
