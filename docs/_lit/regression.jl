

using Pkg; Pkg.activate(joinpath(@__DIR__, "docs"))
using DeadBird
using Distributions, Turing, CSV, DataFrames, NewickTree, Optim, Parameters
using Plots, StatsPlots, Measures
using StatsFuns, SpecialFunctions, ThreadTools, StatsBase

dir = "/home/arzwa/research/gene-family-evolution/data/oryzinae"
data = CSV.read("$dir/nlr-N0-oib.6taxa-counts.csv", DataFrame)
tree = readnw(readline(joinpath(dir, "oryzinae.6taxa.nw")))
covr = CSV.read("$dir/oryzinae.features.csv", DataFrame)
data = innerjoin(data, covr, on=:HOG)


@model simple(Y, model) = begin
    λ ~ Turing.FlatPos(0.)
    μ ~ Turing.FlatPos(0.)
    η ~ Beta()
    ζ ~ Exponential(1.)
    θ = ConstantDLG(λ, μ, typeof(λ)(1e-10))
    p = ShiftedBetaGeometric(η, ζ+1.)
    Y ~ model(rates=θ, rootp=p)
end

dag, bound = CountDAG(data, tree)
model = PhyloBDP(ConstantDLG(λ=0.1, μ=0.1, κ=0.), 
                 ShiftedBetaGeometric(0.71, 3.), 
                 tree, bound)
chn = sample(simple(dag, model), NUTS(), 200)


X = standardize(ZScoreTransform, Matrix(data[:,8:9]), dims=1)

@model regression1(Y, X, model) = begin
    n, m = size(X)
    r1 ~ Normal(log(0.22), 2.)
    r2 ~ Normal(log(0.61), 2.)
    β ~ MvNormal(m, 1.)
    γ ~ MvNormal(m, 1.)
    λ = exp.(r1 .+ X*β)
    μ = exp.(r2 .+ X*γ)
    κ = eltype(λ)(1e-10)
    M = tmap(i->model(rates=ConstantDLG(λ=λ[i], μ=μ[i], κ=κ),
                      bound=Y[i].x[1]), 1:n)
    Y ~ ModelArray(M)
end

rootprior = ShiftedBetaGeometric(0.82, 3.4)
mat, bound = ProfileMatrix(data, tree)
model = PhyloBDP(ConstantDLG(λ=0.1, μ=0.1, κ=0.), rootprior, tree, bound)

chn2 = sample(regression1(mat, X, model), NUTS(), 500)

# GC could be positively correlated with loss rate (higher mutation rate,
# higher pseudogenization)
# gene length could be positively correlated with dup rate and loss rate
# (higher pr to be duplicated)
