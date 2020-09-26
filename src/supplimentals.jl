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


# *******************************************
#  Supplimental Tools
# *******************************************
function xlens_from_xranges(xranges)
    return (x -> -(-(x...))).(xranges)
end
