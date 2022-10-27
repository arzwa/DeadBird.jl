# # DeadBird documentation

# ## Model structure

# The main object of in this package is the `PhyloBDP` model, which bundles 
# - a phylogenetic tree
# - a specification of the duplication, loss and gain rates across the tree
# - a prior on the number of lineages at the root

# In addition, the `PhyloBDP` model object requires the bound on the number 
# of lineages at the root that leave observed descendants. This bound is 
# determined by the data, and is returned by the functions that read in 
# data in `DeadBird`.

using DeadBird, NewickTree, DataFrames, Distributions

# First the data side of things:
data = DataFrame(:A=>[1,2], :B=>[0,1], :C=>[3,3])
tree = readnw("((A:1.0,B:1.0):0.5,C:1.5);")
dag, bound = CountDAG(data, tree)

# Now we specify the model
rates = ConstantDLG(λ=0.5, μ=0.4, κ=0.0)
prior = ShiftedGeometric(0.9)
model = PhyloBDP(rates, prior, tree, bound)

# The model allows likelihood based-inference
loglikelihood(model, dag)

using ForwardDiff

g(x) = loglikelihood(model(rates=ConstantDLG(λ=x[1], μ=x[2])), dag)
ForwardDiff.gradient(g, [0.3, 0.2])

# ## Data structures 
# There are two main data structures to represent the count data. 
#
# (1) There is the `CountDAG`, which efficiently reduces the data to minimize
# the required computations when all families (rows) share the same model
# parameters.
dag, bound = CountDAG(data, tree)

# (2) There is the `ProfileMatrix`, which can be used when model parameters are
# different across families (rows).
mat, bound = ProfileMatrix(data, tree)

# Both give identical results
loglikelihood(model, dag) == loglikelihood(model, mat)

# ## Statistical inference
#
# We use [`Turing.jl`](https://turing.ml/).
using Turing, Optim, Plots, StatsPlots
