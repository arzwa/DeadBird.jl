
using DeadBird
using Distributions, Turing, CSV, DataFrames, NewickTree, Parameters
using Plots, StatsPlots
using Random; Random.seed!(761);

# Load the data
datadir = joinpath(@__DIR__, "../../example/orthofinder")
tree = readnw(readline(joinpath(datadir, "SpeciesTree_rooted.txt")))
data = CSV.read(joinpath(datadir, "Orthogroups.tsv"), DataFrame);

# Filter the data to only deal with families that have members in both clades
# stemming from the root. `cols` will be the set of species in names found in
# the data drame that are also found in the tree.
data, cols = DeadBird.getoib(data, tree)

# The data set size and number of taxa are
n, m = nrow(data), length(getleaves(tree))

# Don't want to deal with huge families. 
Nmax = 10
idx = filter(i->all(Array(data[i,cols]) .<= Nmax), 1:n)
idx = idx[1:10:end]  # subsample
data = data[idx,:]

# The average number of genes in non-extinct families is
avg = mean(filter(x->x>0,Matrix(data[:,cols])))
w = 10  # weight of prior information...
rootprior = ShiftedBetaGeometric(1/avg, 2.)

# We will use the DAG data structure (most efficient, but admits no
# family-specific models).
dag, bound = CountDAG(data[:,cols], tree)

M = PhyloBDP(ConstantDLG(λ=1., μ=1., κ=0.), rootprior, tree, bound)

# We will define a Turing model for this simple problem
@model simplemodel(dag, M, w, avg) = begin
    η ~ Beta(w/avg, w-w/avg)
    ζ ~ Exponential()
    λ ~ Exponential()
    μ ~ Exponential()
    rootprior = ShiftedBetaGeometric(η, ζ+1)
    dag ~ M(rates=ConstantDLG(λ=λ, μ=μ, κ=zero(λ)), rootp=rootprior)
end

# Get a sample from the posterior
chain = sample(simplemodel(dag, M, w, avg), NUTS(), 200)

# check trace plots
plot(chain)

# check model fit using posterior predictive simulations
function mfun(x, M) 
    @unpack λ, μ, η, ζ = DeadBird.getparams(x)
    rootprior = ShiftedBetaGeometric(η, ζ+1)
    M(rates=ConstantDLG(λ=λ, μ=μ, κ=zero(λ)), rootp=rootprior)
end
pp = DeadBird.simulate(y->mfun(y, M), data[:,cols], chain, 1000)
plot(pp)

mat, _ = ProfileMatrix(data[:,cols], tree)
Xs = DeadBird.sample_ancestral(x->mfun(x, M), chain, mat, 100)

Ys = map(i->DeadBird.simulate(mfun(chain[i], M), 1000), 1:length(chain))
Ys = cat(map(y->permutedims(Matrix(y[:,1:2m-1])), Ys)..., dims=3)

# signal low probability families based on Ys -- Xs comparison
# signal low-probability transitions by comparing sampled transitions with BDP
# probabilities.

for n in M.order
    isroot(n) && continue
    a = id(parent(n))
    b = id(n)
    Xs[a,:,:]
    Xs[b,:,:]
end

collect(zip(Xs[8,:,:], Xs[9,:,:]))

