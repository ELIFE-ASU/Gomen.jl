using Documenter
using Gomen

DocMeta.setdocmeta!(Gomen, :DocTestSetup, :(using Gomen); recursive=true)
makedocs(
    sitename = "Gomen",
    format = Documenter.HTML(),
    modules = [Gomen],
    authors = "Douglas G. Moore",
    pages = Any[
        "Home" => "index.md",
    ]
)

deploydocs(
    repo = "github.com/ELIFE-ASU/Gomen.jl.git"
)
