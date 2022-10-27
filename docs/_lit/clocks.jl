# # Some examples of different duplication/loss clock models 
using DataFrames, CSV, Turing, Distributions, NewickTree
using DeadBird
using DeadBird: BrownianLogClock, MvBrownianLogClock

datadir = joinpath(@__DIR__, "../../example/dicots")
tree = readnw(readline(joinpath(datadir, "9dicots.nw")))
data = CSV.read(joinpath(datadir, "9dicots-f01-25.csv"), DataFrame)
const ETA = 1/mean(filter(x->x>0, Matrix(data)))

dag, bound = CountDAG(data, tree)

# ## Critical model
# Here we assume `λ = μ` for each branch. First we use an **uncorrelated relaxed
# clock model**, which is simply a multivariate normal distribution on the
# log-rates.

@model ucr(dag, tree, bound, n=length(postwalk(tree))) = begin
    σ = 0.5
    λ0 ~ Normal(-2, 5)
    λs ~ MvNormal(fill(λ0, n-1), σ)
    λ = [λ0; λs]
    p = ShiftedGeometric(ETA)
    dag ~ PhyloBDP(DLG(λ=λ, μ=λ, κ=zeros(n)), p, tree, bound)
end

chain_ucr = sample(ucr(dag, tree, bound), NUTS(), 100)

# Now we use a model with **autocorrelation** across branches in the tree.
# This is the geometric Brownian motion model with drift correction as used in
# the work of Thorne & Kishino. This uses the formulation where each branch
# rate corresponds to the rate for the midpoint of each branch as in Yang &
# Rannala (2007).
@model gbm(dag, tree, bound) = begin
    σ = 0.5
    λ0 ~ Normal(-2, 5)
    λs ~ BrownianLogClock(λ0, σ, tree)
    λ = [λ0; λs]
    p = ShiftedGeometric(ETA)
    dag ~ PhyloBDP(DLG(λ=λ, μ=λ, κ=zeros(length(λ))), p, tree, bound)
end

chain_gbm = sample(gbm(dag, tree, bound), NUTS(), 100)

# Now we compare
l1 = mean(chain_gbm).nt[2]
l2 = mean(chain_ucr).nt[2]
scatter(l1, l2, grid=false, color=:white, legend=false, size=(330,300), 
        xlabel="autocorrelated", ylabel="uncorrelated")


# ## Two-rates models
# We can use a **multivariate autocorrelated branch-rates model**

@model mvgbm(dag, tree, bound) = begin
    s = 0.1
    ρ ~ Uniform(-0.99, 0.99)
    Σ = [s s*ρ; s*ρ s] 
    r0 ~ Normal(-2, 5)
    rs ~ MvBrownianLogClock([r0, r0], Σ, tree)
    λ = [r0 ; rs[:,1]]
    μ = [r0 ; rs[:,2]]
    p = ShiftedGeometric(ETA)
    dag ~ PhyloBDP(DLG(λ=λ, μ=μ, κ=zeros(length(λ))), p, tree, bound)
end

chain_mvgbm = sample(mvgbm(dag, tree, bound), NUTS(), 100)


