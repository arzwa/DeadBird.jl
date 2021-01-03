# # Maximum likelihood and Bayesian inference for the 12 Drosophila data set

# Here I illustrate the usage of the DeadBird package for fitting phylogenetic
# birth-death process models to data using Maximum likelihood and Bayesian
# inference. We will fit a simple single-rate (turnover rate λ, as in e.g.
# CAFE) model to the 12 *Drosophila* species data set.

# Load the required packages
using DeadBird, Distributions, Turing, CSV, DataFrames, NewickTree, Optim

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

# For the complete data set, this takes a bot 10 seconds. 

# It is straightforward to adapt the model definition to allow for different
# duplication and loss rates, non-zero gain rates (`κ`) or different root
# priors. 

# Now we'll perform Bayesian inference using the No-U-turn sampler. Note that
# we've defined an uninformative flat prior, so we expect to find a posterior 
# mean estimate for `λ` that coincides with the MLE.
chain = sample(model, NUTS(), 100)

# Of course, it would be better to run such a chain for more iterations, e.g. 
# 1000, but for the sake of time I'm only taking a 100 samples here. The 95%
# uncertainty interval for the turnover rate can be obtained as
quantile(chain; q=[0.025, 0.975])
