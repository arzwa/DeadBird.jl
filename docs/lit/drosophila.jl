# # *Drosophila*

# Here I illustrate the usage of the DeadBird package for fitting phylogenetic
# birth-death process models to data using **Maximum likelihood** and **Bayesian
# inference**. We will fit a simple single-rate (turnover rate λ, as in e.g.
# CAFE) model to the 12 *Drosophila* species data set.

# Load the required packages
using DeadBird
using Distributions, Turing, CSV, DataFrames, NewickTree, Optim
using Random; Random.seed!(671);

# Load the data
datadir = joinpath(@__DIR__, "../../example/drosophila")
tree = readnw(readline(joinpath(datadir, "tree.nw")))
data = CSV.read(joinpath(datadir, "counts-oib.csv"), DataFrame);

# The data set size and number of taxa are
nrow(data), length(getleaves(tree))

# We'll take a subset of the data for the sake of time.
data = data[20:10:10010,:];
first(data, 5)

# The average number of genes in non-extinct families is
m = mean(filter(x->x>0,Matrix(data)))

# We can use this to parameterize the prior for the number of ancestral
# lineages
η = 1/m
rootprior = ShiftedGeometric(η)

# We will use the DAG data structure (most efficient).
dag, bound = CountDAG(data, tree)

# We will define a Turing model for this simple problem
@model singlerate(dag, bound, tree, rootprior) = begin
    λ ~ Turing.FlatPos(0.)
    θ = ConstantDLG(λ=λ, μ=λ, κ=zero(λ))
    dag ~ PhyloBDP(θ, rootprior, tree, bound)
end

# ## Maximum likelihood inference
model = singlerate(dag, bound, tree, rootprior)
@time mleresult = optimize(model, MLE())

# For the complete data set, this takes about 10 seconds. 

# It is straightforward to adapt the model definition to allow for different
# duplication and loss rates, non-zero gain rates (`κ`) or different root
# priors. 

# ## Bayesian inference
# Now we'll perform Bayesian inference using the No-U-turn sampler. Note that
# we've defined an uninformative flat prior (`FlatPos(0.0)`), so we expect to
# find a posterior mean estimate for `λ` that coincides with the MLE.
chain = sample(model, NUTS(), 100)

# Of course, it would be better to run such a chain for more iterations, e.g. 
# 1000, but for the sake of time I'm only taking a 100 samples here. The 95%
# uncertainty interval for the turnover rate can be obtained as
quantile(chain; q=[0.025, 0.975])

# ## Other models
# It is straightforward to use the [Turing.jl](https://turing.ml/dev/) model
# syntax (using the `@model` macro) in combination with the various rates
# models and root priors defined in DeadBird (`ConstantDLG`, `DLG`, `DLGWGM`, 
# `ShiftedBetaGeometric`, ...) to specify complicated models. A not so 
# complicated example would be the following. First we filter the data to only
# allow for non-extinct families:
nonextinct = filter(x->all(Array(x) .> 0), data);

# We will model the excess number of genes, i.e. the number of extra
# (duplicated) genes *per* family, instead of the total number of genes. 
excessgenes = nonextinct .- 1;

# Again we construct a DAG object
dag, bound = CountDAG(excessgenes, tree)

# The model we specify is a linear birth-death and immigration process with
# immigration (gain) rate equal to the duplication rate, `κ = λ`, and loss rate
# `μ`. This corresponds to a model where genes duplicate at rate λ, (note that
# a `0 -> 1` transition is also a duplication here since the zero state
# corresponds to a single copy family), and where *duplicated genes* get lost
# at rate `μ`. We assume `λ < μ`, in which case there is a geometric stationary
# distribution with mean `1 - λ/μ` for the excess number of genes in a family.
bound01(η) = η <= zero(η) ? zero(η) + 1e-16 : η >= one(η) ? one(η) - 1e-16 : η

@model nonextinctmodel(dag, bound, tree) = begin
    μ ~ Turing.FlatPos(0.)
    η ~ Beta(1, 1)  # 1 - λ/μ
    η = bound01(η)  
    λ = μ * (1 - η)
    rates = ConstantDLG(λ=λ, μ=μ, κ=λ)
    rootp = Geometric(η)
    dag ~ PhyloBDP(rates, rootp, tree, bound, cond=:none)
end

# and we sample
chain = sample(nonextinctmodel(dag, bound, tree), NUTS(), 100)

# Get the posterior as a dataframe
pdf = DataFrame(chain)
μs = pdf[:, :μ]
λs = μs .* (1 .- pdf[:,:η])

# The marginal posterior mean duplication rate (and 95% uncertainty interval) is
mean(λs), quantile(λs, [0.025, 0.975])

# The marginal posterior mean loss rate *per duplicated gene* is
mean(μs), quantile(μs, [0.025, 0.975])
