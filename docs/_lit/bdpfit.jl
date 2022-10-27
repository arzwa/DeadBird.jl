using DeadBird
using Distributions, Turing, CSV, DataFrames, NewickTree, Optim, Parameters
using Plots, StatsPlots, StatsFuns, SpecialFunctions
using Plots: mm
datadir = joinpath(@__DIR__, "../data/")

# Rice example
# ============
data = CSV.read(joinpath(datadir, "rice/oryza-max10-6taxa-oib.csv"), DataFrame)
tree = readnw(readline(joinpath(datadir, "rice/oryzinae.6taxa.nw")))
taxa = Dict(x=>x[1][1] * ". " * join(split(x, "_")[2:end], " ") 
            for x in name.(getleaves(tree)))  # prettier labels

# We will use the DAG data structure (most efficient, but admits no
# family-specific models).
dag, bound = CountDAG(data, tree)

# Linear BDP
# ----------
#
# To get an idea of the order of magnitude of the rates, we'll do ML estimation
# with an empirical prior distribution for the ancestral family size.
η̂ = 1/mean(Matrix(data))

@model mlefit(dag, bound, tree, η, ::Type{T}=Float64) where T = begin
    λ ~ Exponential(1.)
    μ ~ Exponential(1.)
    θ = ConstantDLG(λ=λ, μ=μ) 
    p = ShiftedGeometric(T(η))
    dag ~ PhyloBDP(θ, p, tree, bound)
end

optimize(mlefit(dag, bound, tree, η̂), MLE(), [0.1, 0.1])

# Now for Bayesian inference
@model constantrate(dag, bound, tree) = begin
    η ~ Beta()
    ζ ~ Exponential(1.)
    λ ~ Exponential(0.2)
    μ ~ Exponential(0.2)
    θ = ConstantDLG(λ=λ, μ=μ)
    p = ShiftedBetaGeometric(η, ζ+1)
    dag ~ PhyloBDP(θ, p, tree, bound)
end

chn1 = sample(constantrate(dag, bound, tree), NUTS(), 500)

# Posterior predictive checks
function mfun1(x, tree, bound)
    @unpack λ, μ, η, ζ = DeadBird.getparams(x)
    θ = ConstantDLG(λ=λ, μ=μ, κ=0.)
    p = ShiftedBetaGeometric(η, ζ)
    PhyloBDP(θ, p, tree, bound)
end

pp1 = DeadBird.simulate(y->mfun1(y, tree, bound), data, chn1, 1000)
p1 = plot(pp1, xscale=:identity, xlim=(1,11), xticks=(1:11, 0:10))


# BDIP model for excess genes
# ---------------------------
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
    μ ~ Exponential()
    η ~ Beta(1, 1)  # 1 - λ/μ
    ζ ~ Exponential()
    η = bound01(η)  
    λ = μ * (1 - η)
    rates = ConstantDLG(λ=λ, μ=μ, κ=λ)
    rootp = BetaGeometric(η, 1 + ζ)
    dag ~ PhyloBDP(rates, rootp, tree, bound, cond=:none)
end

# and we sample
chn2 = sample(nonextinctmodel(dag, bound, tree), NUTS(), 500)

# get λ posterior
λs = (1 .- chn2[:η]) .* chn2[:μ]
Chains(λs)

# posterior prediction
function mfun2(x, tree, bound) 
    @unpack η, ζ, μ = DeadBird.getparams(x)
    λ = μ * (1 - η)
    p = BetaGeometric(η, 1 + ζ)
    θ = ConstantDLG(λ=λ, μ=μ, κ=λ)
    M = PhyloBDP(θ, p, tree, bound, cond=:none)
end

pp2 = DeadBird.simulate(y->mfun2(y, tree, bound), excessgenes, chn2, 1000)

p1 = plot(pp1, taxa=taxa, xscale=:identity, xlim=(1,11), xticks=(1:11, 0:10), xlabel="", ylabel="")
p2 = plot(pp2, taxa=taxa, color=:salmon, xscale=:identity, xlim=(1,11), xticks=(1:11, 0:10), xlabel="", ylabel="")
for p in [p1, p2]
    xlabel!.(p.subplots[4:end], "\$n\$")
    for sp in p.subplots[4:end]
        sp.attr[:bottom_margin] = 5mm
    end
end
for sp in p1.subplots[[1,4]]
    ylabel!(sp, "\$\\log_{10}f_n\$")
    sp.attr[:left_margin] = 5mm 
end
plot(p1, p2, size=(1000,250), titlefont=8)

# Discrete Beta mixture
# ---------------------

# Normal approximation to beta, and discretize
expit(x) = exp(x)/(1+exp(x))
function discretebeta(η, ζ, K)
    α = η*ζ
    β = (1-η)*ζ
    m = digamma(α) - digamma(β)
    v = trigamma(α) + trigamma(β)
    ps = expit.(DeadBird.discretize(Normal(m, √v), K))
end

@model nonextinctmodelmix(dag, bound, tree, K=4) = begin
    η ~ Beta(1,1)  # 1 - λ/μ
    ζ ~ Exponential()
    μ ~ Exponential()
    η = bound01(η)  
    α = discretebeta(η, 1+ζ, K)
    λ = μ .* (1 .- α)
    θ = [ConstantDLG(λ=λ[i], μ=μ, κ=λ[i]) for i=1:K]
    p = BetaGeometric(η, 1+ζ)
    dag ~ MixtureModel([PhyloBDP(θ[i], p, tree, bound, cond=:none) for i=1:K])
end

K = 4
dag, bound = CountDAG(excessgenes, tree)
chn3 = sample(nonextinctmodelmix(dag, bound, tree, K), NUTS(), 500)

function mfun3(x, tree, bound, K) 
    @unpack η, ζ, μ = getparams(x)
    λ = μ .* (1 .- discretebeta(η, 1+ζ, K))
    rates = [ConstantDLG(λ=λ[i], μ=μ, κ=λ[i]) for i=1:K]
    rootp = BetaGeometric(η, 1+ζ)
    M = MixtureModel([PhyloBDP(rates[i], rootp, tree, bound, cond=:none) for i=1:K])
end
pp3 = DeadBird.simulate(y->mfun3(y, tree, bound, K), excessgenes, chn3, 1000)
plot(pp3)


# test with simulation whether BDIP/BDP comparison holds
using DeadBird, StatsBase
θ1 = ConstantDLG(λ=0.08, μ=0.34, κ=0.)
θ2 = ConstantDLG(λ=0.08, μ=0.34, κ=0.08)
p1 = ShiftedGeometric(0.91)
M1 = PhyloBDP(θ1, p1, tree, 1, cond=:none)

Y1 = DeadBird.simulate(M1, 10000)
X1 = DeadBird.observedmatrix(Y1, M1)
idx = filter(i->all(Array(X1[i,:])[1:6] .> 0), 1:nrow(X1))
C1 = proportions(Matrix(X1[idx,1:6]))
condprior = proportions(Y1[idx,1])  
# this is the distribution over ancestral family sizes conditional on
# non-extinction, very different from the original Geometric!

p2 = Geometric(condprior[1])
#p2 = Geometric(0.89)
M2 = PhyloBDP(θ2, p2, tree, 1, cond=:none)
X2 = DeadBird.observedmatrix(DeadBird.simulate(M2, 10000), M2)
X2 = X2 .+ 1
X2 = X2[1:nrow(X1),:]
C2 = proportions(Matrix(X2[:,1:6]))
C1

