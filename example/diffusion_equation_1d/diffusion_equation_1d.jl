using Plots
using SpectralMethodFFT


function x_k_init(c::ConfigFFT)

    x_X = x_Xgen(c)
    x_u = sinc(x_X)
    k_u = k_x(x_u)
    return x_u, k_u

end


function k_develop_k(k_u::KFunc, dt::Real)

    c = k_u.config
    xrange = c.xranges[1]
    xlen = xrange[2] - xrange[1]
    k_K = k_Kgen(c)

    return exp(- (2π*k_K/xlen)^2 * dt) * k_u

end


function main()

    # -------------
    # Parameters
    # -------------
    # *** space ***
    xrange = (-100., 100.)
    ngrid = 128
    c = ConfigFFT((ngrid,), (xrange,))
    # *** time ***
    t_st = 0.
    t_ed = 0.1
    nt = 300
    dt = (t_ed - t_st) / nt

    # -------------
    # Time Integration
    # -------------
    x_u, k_u = x_k_init(c)
    x_X = x_Xgen(c)

    anim = Animation()
    for it = 0:nt
        x_u = x_k(k_u)
        plt = plot(x_X.vals, x_u.vals, ylims=(-0.2, 1.2))
        frame(anim, plt)
        if it < nt
            k_u = k_develop_k(k_u, dt)
        end
    end
    gif(anim, "heat_1d.gif", fps=3)

end

main()
