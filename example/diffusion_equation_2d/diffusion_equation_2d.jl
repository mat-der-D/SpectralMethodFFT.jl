using Plots
using Printf
using SpectralMethodFFT


"""
    xy_kl_init(c::ConfigFFT)

Impose initial condition as sinc function

    ```
    u(x,y) = sinc(\sqrt{x^2+y^2})
    ```
"""
function xy_kl_init(c::ConfigFFT)

    xy_X = xy_Xgen(c)
    xy_Y = xy_Ygen(c)

    xy_r = sqrt(xy_X^2 + xy_Y^2)
    xy_u = sinc(xy_r / 2)
    kl_u = kl_xy(xy_u)

    return xy_u, kl_u

end


"""
    kl_develop_kl(k_u::KFunc, dt::Real)

Compute next time step of diffusion equation

    ```
    u^{n+1} = exp(-4π^2 Δt ((kx/Lx)^2 + (ky/Ly)^2) ) u^n
    ```
"""
function kl_develop_kl(kl_u::KFunc, dt::Real)

    c = kl_u.config
    xlen, ylen = xlens_xranges(c.xranges)

    kl_K = kl_Kgen(c)
    kl_L = kl_Lgen(c)

    return exp( - 4 * π^2 * dt * (
        (kl_K/xlen)^2 + (kl_L/ylen)^2
    )) * kl_u

end


function main()

    # -------------
    # Parameters
    # -------------
    # *** space ***
    xranges = ((-10., 10.), (-10., 10.))
    ngrids = (128, 128)
    c = ConfigFFT(ngrids, xranges)
    # *** time ***
    t_st = 0.
    t_ed = 2.0
    nt = 200
    dt = (t_ed - t_st) / nt

    # -------------
    # Time Integration
    # -------------
    xy_u, kl_u = xy_kl_init(c)
    xy_X = xy_Xgen(c)
    xy_Y = xy_Ygen(c)

    anim = Animation()
    for it = 0:nt
        t = @sprintf("%3.2f", t_st + it*dt)
        xy_u = xy_kl(kl_u)
        plt = plot(
                xy_X[:,1], xy_Y[1,:], xy_u,
                title="t=$t",
                zlims=(-0.25, 1.1),
                xlabel="x", ylabel="y", zlabel="u",
                st=:wireframe)
        frame(anim, plt)
        if it < nt
            kl_u = kl_develop_kl(kl_u, dt)
        end
    end
    gif(anim, "heat_2d.gif", fps=20)

end


main()
