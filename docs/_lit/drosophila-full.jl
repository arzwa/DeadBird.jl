# Load the required packages
using DeadBird
using Distributions, Turing, CSV, DataFrames, NewickTree, Optim
using Random; Random.seed!(761);

# Load the data
#datadir = joinpath(@__DIR__, "../../example/drosophila")
datadir = joinpath(@__DIR__, "example/drosophila")
tree = readnw(readline(joinpath(datadir, "tree.nw")))
data = CSV.read(joinpath(datadir, "counts-oib.csv"), DataFrame);

data = filter(x->all(Array(x) .≤ 10), data)

# The data set size and number of taxa are
nrow(data), length(getleaves(tree))

# The average number of genes in non-extinct families is
m = mean(filter(x->x>0,Matrix(data)))

# We will use the DAG data structure (most efficient, but admits no
# family-specific models).
dag, bound = CountDAG(data, tree)

# We will define a Turing model for this simple problem
@model singlerate(dag, bound, tree, rootprior) = begin
    λ ~ Turing.FlatPos(0.)
    θ = ConstantDLG(λ=λ, μ=λ, κ=zero(λ))
    dag ~ PhyloBDP(θ, rootprior, tree, bound)
end

@model model1(dag, bound, tree, ::Type{T}=Float64) where T = begin
    η ~ Beta()
    ζ ~ Turing.FlatPos(1.)
    μ ~ Exponential()
    λ ~ Exponential()
    θ = ConstantDLG(λ=λ, μ=λ, κ=T(1e-10))
    p = ShiftedBetaGeometric(η, ζ)
    dag ~ PhyloBDP(θ, p, tree, bound)
end

dro1 = sample(model1(dag, bound, tree), NUTS(), 200)

