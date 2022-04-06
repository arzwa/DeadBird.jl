# Load the required packages
using DeadBird
using Distributions, Turing, CSV, DataFrames, NewickTree, Optim
using Random; Random.seed!(761);

# Load the data
#datadir = joinpath(@__DIR__, "../../example/drosophila")
datadir = joinpath(@__DIR__, "example/drosophila")
tree = readnw(readline(joinpath(datadir, "tree.nw")))
data = CSV.read(joinpath(datadir, "counts-oib.csv"), DataFrame);

# The data set size and number of taxa are
nrow(data), length(getleaves(tree))

# We'll take a subset of the data for the sake of time.
#data = data[20:10:10010,:];
#first(data, 5)

# The average number of genes in non-extinct families is
m = mean(filter(x->x>0,Matrix(data)))

# We can use this to parameterize the prior for the number of ancestral
# lineages
η = 1/m
rootprior = ShiftedGeometric(η)

# We will use the DAG data structure (most efficient, but admits no
# family-specific models).
dag, bound = CountDAG(data, tree)

# We will define a Turing model for this simple problem
@model singlerate(dag, bound, tree, rootprior) = begin
    λ ~ Turing.FlatPos(0.)
    θ = ConstantDLG(λ=λ, μ=λ, κ=zero(λ))
    dag ~ PhyloBDP(θ, rootprior, tree, bound)
end

# ## Maximum likelihood inference
#
# First we show how to conduct MLE of a single parameter model for the entire
# data (i.e. we estimate a genome-wide parameter) using the `CountDAG` data
# structure.
model = singlerate(dag, bound, tree, rootprior)
@time mleresult = optimize(model, MLE())

# For the complete data set of >10000 families, this takes about 10 seconds. 

# It is straightforward to adapt the model definition to allow for different
# duplication and loss rates, non-zero gain rates (`κ`) or different root
# priors. 

# Alternatively we could use the ProfileMatrix, which admits models that deal
# with variation across families. We can also use this to fit models
# independently across families.
# Here we will estimate the MLE of a single turnover rate for 100 families
# independently.
matrix, bound = ProfileMatrix(data, tree)

@model singlerate(mat, bound, tree, rootprior) = begin
    λ ~ Turing.FlatPos(0.)
    θ = ConstantDLG(λ=λ, μ=λ, κ=zero(λ))
    mat ~ PhyloBDP(θ, rootprior, tree, bound)
end

@time results = map(1:size(matrix, 1)) do i
    x = matrix[i]
    model = singlerate(x, x.x[1], tree, rootprior)
    mleresult = optimize(model, MLE())
    mleresult.lp, mleresult.values[1]
end;
