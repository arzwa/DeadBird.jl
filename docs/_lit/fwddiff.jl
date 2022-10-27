using Pkg; Pkg.activate(joinpath(@__DIR__, "docs"))
using DeadBird
using CSV, DataFrames, NewickTree, Optim, Parameters, BenchmarkTools
using ForwardDiff

dir = "docs/data/dicots"
data = CSV.read("$dir/9dicots-f01-1000.csv", DataFrame)
tree = readnw(readline(joinpath(dir, "9dicots.nw")))
dag, bound = CountDAG(data, tree)

n = length(postwalk(tree))
θ = DLG(λ=randn(n), μ=randn(n), κ=zeros(n))
M = PhyloBDP(θ, ShiftedGeometric(0.9), tree, bound)

function lfun(x)
    θ = DLG(λ=x[1:n], μ=x[n+1:end], κ=zeros(n))
    logpdf(M(rates=θ), dag)
end
        
@benchmark ForwardDiff.gradient(lfun, randn(2n))

@profile ForwardDiff.gradient(lfun, randn(2n))
