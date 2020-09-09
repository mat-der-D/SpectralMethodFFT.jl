# fft version
using FFTW


# *******************************************
#  Configuration
# *******************************************
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
            ) where N where T
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
mutable struct XFunc{T,N} <: AbstractArray{T,N}

    vals::Array{T,N}
    config::ConfigFFT{T,N}

    # CONSTRUCTOR
    function XFunc{T,N}(
                vals::Array{T,N},
                config::ConfigFFT{T,N}
            ) where N where T

        if size(vals) == config.ngrids
            return new(vals, config)
        else
            println("ERROR")
        end

    end

    # EASY CONSTRUCTOR
    function XFunc(
                vals::Array{Tv,N},
                config::ConfigFFT{Tc,N}
            ) where N where Tv <: Number where Tc

        XFunc{Tc,N}(Tc.(vals), config)

    end

    # UNDEF CONSTRUCTOR
    function XFunc(
                undef::UndefInitializer,
                config::ConfigFFT{T,N}
                ) where N where T

        f_undef = Array{T,N}(undef, config.ngrids)
        return XFunc{T,N}(f_undef, config)

    end

end

Base.:size(f::XFunc) = size(f.vals)

Base.:getindex(f::XFunc, i::Int) = getindex(f.vals, i)
function Base.:getindex(
            f::XFunc{T,N}, I::Vararg{Int,N}
        ) where N where T
    getindex(f.vals, I...)
end

Base.:setindex!(f::XFunc, v, i::Int) = setindex!(f.vals, v, i)
function Base.:setindex!(
            f::XFunc{T,N}, v, I::Vararg{Int,N}
        ) where N where T
    setindex!(f, v, I...)
end

Base.:copy(f::XFunc) = XFunc(copy(f.vals), f.config)


# *******************************************
#  KFunc
# *******************************************
mutable struct KFunc{T,N} <: AbstractArray{Complex{Float64},N}

    vals::Array{Complex{Float64},N}
    config::ConfigFFT{T,N}

    # CONSTRUCTOR
    function KFunc{T,N}(
                vals::Array{Complex{Float64},N},
                config::ConfigFFT{T,N}
            ) where N where T

        if size(vals) == config.ngrids
            return new(vals, config)
        else
            println("ERROR")
        end

    end

    # EASY CONSTRUCTOR
    function KFunc(
                vals::Array{Tv,N},
                config::ConfigFFT{Tc,N}
            ) where N where Tv <: Number where Tc
        KFunc{Tc,N}(complex(float(vals)), config)
    end

    # UNDEF CONSTRUCTOR
    function KFunc(
                undef::UndefInitializer,
                config::ConfigFFT{T,N}
            ) where N where T

        f_undef = Array{Complex{Float64},N}(undef, config.ngrids)
        return KFunc{T,N}(f_undef, config)

    end

end

Base.:size(f::KFunc) = size(f.vals)

Base.:getindex(f::KFunc, i::Int) = getindex(f.vals, i)
function Base.:getindex(
            f::KFunc{T,N}, I::Vararg{Int,N}
        ) where N where T
    getindex(f.vals, I...)
end

Base.:setindex!(f::KFunc, v, i::Int) = setindex!(f.vals, v, i)
function Base.:setindex!(
            f::KFunc{T,N}, v, I::Vararg{Int,N}
        ) where N where T
    setindex!(f, v, I...)
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
            if f.config === g.config
                return XFunc($opd(f.vals, g.vals), f.config)
            else
                println("ERROR")
            end
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
            if f.config == g.config
                return KFunc($opd(f.vals, g.vals), f.config)
            else
                println("ERROR")
            end
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
        ) where N where T

    vals = copy(f.vals)
    vals[slices...] .= 0.0 + 0.0im
    f.vals -= vals

end

# +++++ high-pass filter +++++
function highpass_K!(
            f::KFunc{T,N},
            min_nwaves::NTuple{N,Int}
        ) where N where T
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
        ) where N where T

    g = copy(f)
    highpass_K!(g, min_nwaves)
    return g

end

# +++++ low-pass filter +++++
function lowpass_K!(
            f::KFunc{T,N},
            max_nwaves::NTuple{N,Int}
        ) where N where T

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
        ) where N where T

    g = copy(f)
    lowpass_K!(g, max_nwaves)
    return g

end


# *******************************************
#  Fourier Transformation
# *******************************************
function K_X(f::XFunc{T,N}) where T where N

    P = f.config.P_fft
    g = KFunc(P * f.vals, f.config)

    if f.config.cut_zigzag_mode
        ngrids = f.config.ngrids
        max_nwaves = @. ngrids ÷ 2 - 1
        lowpass_K!(g, max_nwaves)
    end

    return g
end

function X_K(f::KFunc{T,N}) where N where T
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

function K_laplacian_K(f::KFunc{T,N} where T where N)
    return sum(
        K_∂Xaxis_K(K_∂Xaxis_K(f, axis), axis)
        for axis = 1:N
    )
end

K_Δ_K = K_laplacian_K
X_laplacian_X = X_K ∘ K_laplacian_K ∘ K_X
X_Δ_X = X_laplacian_X
