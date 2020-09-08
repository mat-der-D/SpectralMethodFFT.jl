using SpectralMethodFFT
using Documenter

makedocs(;
    modules=[SpectralMethodFFT],
    authors="Smooth Pudding",
    repo="https://github.com/mat-der-D/SpectralMethodFFT.jl/blob/{commit}{path}#L{line}",
    sitename="SpectralMethodFFT.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://mat-der-D.github.io/SpectralMethodFFT.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/mat-der-D/SpectralMethodFFT.jl",
)
