using Plots
using Printf
using SpectralMethodFFT


function x_k_init(c::ConfigFFT)

    ngrid = c.ngrids[1]
    ix_mid = ngrid÷2 + 1

    x_u = XFunc(zeros(ngrid), c)
    x_u[ix_mid] = 1.
    k_u = k_x(x_u)
    return x_u, k_u

end


function k_develop_k(k_u::KFunc, dt::Real)

    c = k_u.config
    xlen, = xlens_from_xranges(c.xranges)
    k_K = k_Kgen(c)

    return exp(- (2π*k_K/xlen)^2 * dt) * k_u

end


function main()

    # -------------
    # Parameters
    # -------------
    # *** space ***
    xrange = (-10., 10.)
    ngrid = 128
    c = ConfigFFT((ngrid,), (xrange,))
    # *** time ***
    t_st = 0.
    t_ed = 1.
    nt = 200
    dt = (t_ed - t_st) / nt

    # -------------
    # Time Integration
    # -------------
    x_u, k_u = x_k_init(c)
    x_X = x_Xgen(c)

    anim = Animation()
    for it = 0:nt
        t = @sprintf("%4.3f", t_st + it*dt)
        x_u = x_k(k_u)
        plt = plot(
                x_X, x_u,
                title="t=$t",
                label="heat",
                ylims=(-0.1, 1.1),
                xlabel="x", ylabel="u")
        frame(anim, plt)
        if it < nt
            k_u = k_develop_k(k_u, dt)
        end
    end
    gif(anim, "heat_1d.gif", fps=20)

end

main()
