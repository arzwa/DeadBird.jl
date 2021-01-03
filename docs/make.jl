using Documenter, DeadBird, Literate
outdir = joinpath(@__DIR__, "src")
srcdir = joinpath(@__DIR__, "lit")
mkpath(outdir)

fnames = ["drosophila.jl", "index.jl"]
output = String[]

for f in fnames
    target = string(split(f, ".")[1])
    outpath = joinpath(outdir, target*".md")
    f != "index.jl" && push!(output, relpath(outpath, joinpath(@__DIR__, "src")))
    @info "Literating $f"
    Literate.markdown(joinpath(srcdir, f), outdir, documenter=true)
    x = read(`tail -n +4 $outpath`)
    write(outpath, x)
end

makedocs(
    modules = [DeadBird],
    sitename = "DeadBird.jl",
    authors = "Arthur Zwaenepoel",
    doctest = false,
    pages = ["Index"    => "index.md",
             "Examples" => output,
             "API"      => "api.md"],)

deploydocs(repo = "github.com/arzwa/DeadBird.jl.git", devbranch="main")

