using Pkg; Pkg.activate(joinpath(@__DIR__, "docs"))
using DeadBird
using Distributions, Turing, CSV, DataFrames, NewickTree, Optim, Parameters
using Plots, StatsPlots, Measures, StatsBase
using Optim, LineSearches

# Primates data -- a bit challenging
dir = "docs/data/CAFE/"
tree = readnw(readline(joinpath(dir, "mammals_tree.txt")))
for n in postwalk(tree)
    n.data.distance /= 100
end
df = CSV.read(joinpath(dir, "mammal_gene_families.txt"), DataFrame)
df = filter(x->sum(Array(x[3:end])) > x["rat"] + x["mouse"], df)[1:10000,:]
CSV.write(joinpath(dir, "mammals-filtered.txt"), df, delim="\t")

# ~/dev/CAFE5/bin/cafe5 -i mammals-filtered.txt -t mammals_tree.txt -c 1 -z -p
# Empirical Prior Estimation Result : (34 iterations)
# Poisson lambda: 0.83421754241661 &  Score: 226853.28848078
# Completed 26 iterations
# Time: 0H 4M 55S
# Best match is: 0.0019338866249132
# Final -lnL: 117337.91326818

dag, bound = CountDAG(df[1:10000,:], tree)
M = PhyloBDP(ConstantDLG(), Poisson(0.83), tree, bound, cond=:none)
@model cafe(M, dag, ::Type{T}=Float64) where T = begin
    λ ~ Uniform(1e-9,10.)
    dag ~ M(rates=ConstantDLG(λ=λ, μ=λ, κ=T(1e-7)))
end
@time res = optimize(cafe(M, dag), MAP(), BFGS(; linesearch = LineSearches.BackTracking()))


# dicots data
dir = "docs/data/dicots"
tree = readnw(readline(joinpath(dir, "9dicots.nw")))
df = CSV.read(joinpath(dir, "9dicots-f01-1000.cafe"), DataFrame)

dag, bound = CountDAG(df, tree)
M = PhyloBDP(ConstantDLG(), Poisson(0.55), tree, bound, cond=:none)
@model cafe(M, dag, ::Type{T}=Float64) where T = begin
    λ ~ Uniform(0,10.)
    dag ~ M(rates=ConstantDLG(λ=λ, μ=λ, κ=T(1e-7)))
end
@time res = optimize(cafe(M, dag), MAP(), BFGS(; linesearch = LineSearches.BackTracking()))

chn = sample(cafe(M, dag), NUTS(), 100)
