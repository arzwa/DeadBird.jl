# BirdDad

# This is a reimplementation of the core functionalities previously implemented in the `Beluga` library.

# ## Maximum-likelihood estimation
using BirdDad, NewickTree, DelimitedFiles, Distributions
using ForwardDiff, Optim
import BirdDad: CountDAG, ConstantDLG, PhyloBDP, mle_problem, RatesModel, DLG

# Read in the data and tree
X, s = readdlm("example/9dicots-f01-1000.csv", ',', Int, header=true)
tree = readnw(readline("example/9dicots.nw"))

# Construct the data object
dag, bound = CountDAG(X, s, tree)

# Construct the model
rates = RatesModel(ConstantDLG(λ=.1, μ=.1, κ=0.0, η=1/1.5),fixed=(:η, :κ))
model = PhyloBDP(rates, tree, bound)

f, ∇f = mle_problem(dag, model)
@time out = optimize(f, ∇f, randn(2), BFGS())
t = BirdDad.transform(model.rates.trans, out.minimizer)

# Note that this is an order of magnitude faster than the `R` implementation in WGDgc (where I measured a run time ≈35s for the same data and initial conditions). It is also *a lot* faster than CAFE.

# ## Side note:
# The Hessian of the negative loglikelihood evluated at the MLE is equal to the observed Fisher information $I(\hat{\theta})$. The estimated standard errors for the ML estimates $\mathrm{SE}(\hat{\theta}) = 1/\sqrt{I(\hat{\theta})}$ can be obtained as follows
using ForwardDiff, LinearAlgebra

function ff(x::Vector{T}) where T
    rates = model.rates((λ=x[1], μ=x[2]))
    m = PhyloBDP(rates, tree, bound)
    d = copydag(dag, T)
    -loglikelihood!(d, m)
end
1 ./ diag(.√(ForwardDiff.hessian(ff, [t.λ, t.μ])))

# Note that, sadly, we cannot use `ForwardDiff.hessian(f, out.minimizer)` to obtain this, since that would calculate the hessian in transformed space.

# ## Bayesian inference using DynamicHMC
using DynamicHMC, LogDensityProblems, Random, DynamicHMC.Diagnostics
using Parameters

# First, we need to code our 'LogDensityProblem' struct with an associated
# function that returns the log posterior density. Note that the problem should
# a subtype of `BirdDad.Problem`
struct ConstantDLGProblem <: BirdDad.Problem
    model
    data
end

function (p::ConstantDLGProblem)(θ)
    @unpack λ, μ, η, κ = θ
    ℓ = loglikelihood(p, θ)
    π = logpdf(MvLogNormal(ones(2)), [λ, μ]) +
         logpdf(Exponential(0.1), κ) +
         logpdf(Beta(3,1), η)
    return ℓ + π
end

X, s = readdlm("example/9dicots-f01-100.csv", ',', Int, header=true)
tree = readnw(readline("example/9dicots.nw"))
dag, bound = CountDAG(X, s, tree)
r = RatesModel(ConstantDLG(λ=1.0, μ=1.2, κ=0.01 , η=1/mean(X)))
m = PhyloBDP(r, tree, bound)
p = ConstantDLGProblem(m, dag)
t = BirdDad.trans(p)
P = TransformedLogDensity(t, p)
∇P = ADgradient(:ForwardDiff, P);
results = mcmc_with_warmup(Random.GLOBAL_RNG, ∇P, 100);
posterior = transform.(t, results.chain)
@info summarize_tree_statistics(results.tree_statistics)

# ## Bayesian inference with Turing
using Turing

# ### Constant rates DLG model
@model constantrates(dag, model) = begin
    r ~ MvLogNormal(ones(2))
    η ~ Beta(3,1)
    κ ~ Exponential(0.01)
    dag ~ model((λ=r[1], μ=r[2], η=η, κ=κ))
end

X, s = readdlm("example/9dicots-f01-100.csv", ',', Int, header=true)
tree = readnw(readline("example/9dicots.nw"))
dag, bound = CountDAG(X, s, tree)
r = RatesModel(BirdDad.ConstantDLG(λ=1., μ=1., κ=0.01 , η=0.66))
m = PhyloBDP(r, tree, bound)
model = constantrates(dag, m)
chain = sample(model, NUTS(0.65), 1000);

# Fun!

@model constantrates(dag, model) = begin
    r ~ MvLogNormal(ones(2))
    η ~ Beta(3,1)
    α ~ Exponential()
    κ ~ Exponential(0.01)
    dag ~ model((λ=r[1], μ=r[2], η=η, κ=κ, α=α))
end

X, s = readdlm("example/9dicots-f01-100.csv", ',', Int, header=true)
tree = readnw(readline("example/9dicots.nw"))
dag, bound = CountDAG(X, s, tree)
r = RatesModel(GammaMixture(ConstantDLG(λ=1., μ=1., κ=0.01 , η=0.66), 4))
m = PhyloBDP(r, tree, bound)
model = constantrates(dag, m)
chain = sample(model, NUTS(0.65), 1000);

# There is an issue with quantiles from the Gamma!

# ### Branch-wise exponential DL 'distances'

# This is a simple model to estimate the expected number of DL events per gene for each branch. This is similar to what is usually done for standard models in sequence-based phylogenetics, i.e. we estimate a distance instead of a rate. I use an uninformative exponential prior corresponding to a mean of 0.5 events per gene for any given branch. This uses the `DLG` (duplication-loss-gain model) with gain parameter κ fixed to 0.0

# Read data
X, s = readdlm("example/9dicots-f01-25.csv", ',', Int, header=true)
tree = readnw(readline("example/9dicots.nw"))

# Set all branch lengths to 1
for n in postwalk(tree) n.data.distance = 1. end
n = length(postwalk(tree))

# Process the data and construct model
dag, bound = CountDAG(X, s, tree)
rates = RatesModel(DLG(λ=ones(n), μ=ones(n), κ=0.0, η=1/mean(X)), fixed=(:κ,))
basemodel = PhyloBDP(rates, tree, bound)

# The Bayesian model
@model branchrates(dag, model, ::Type{T}=Float64) where {T} = begin
    n = length(postwalk(model[1]))
    λ = zeros(T, n)
    η ~ Beta(3,1)
    for i=2:n
        λ[i] ~ Exponential(0.5)
    end
    dag ~ model((λ=λ, μ=λ, η=η))
end

# and now do inference
model = branchrates(dag, basemodel)
chain = sample(model, NUTS(0.65), 1000);

# ### Slightly more involved DL distances
@model branchrates(dag, model, ::Type{T}=Float64) where {T} = begin
    n = length(postwalk(model[1]))
    η ~ Beta(3,1)
    ν ~ Exponential(0.1)
    λ = zeros(T, n)
    μ = zeros(T, n)
    for i=2:n
        λ[i] ~ Exponential(0.5)
        μ[i] ~ LogNormal(log(λ[i]), ν)
    end
    dag ~ model((λ=log.(λ), μ=log.(μ), η=η))
end

model = branchrates(dag, basemodel)
chain = sample(model, NUTS(0.65), 500);

# ### An uncorrelated relaxed DL clock
@model branchrates(dag, model, ::Type{T}=Matrix{Float64}) where {T} = begin
    n = length(postwalk(model[1]))
    η ~ Beta(3,1)
    Σ ~ InverseWishart(3, [1. 0. ; 0. 1.0])
    r = T(undef, 2, n)
    r[:,1] ~ MvNormal(zeros(2), ones(2))
    for i=2:n
        r[:,i] ~ MvNormal(r[:,1], Σ)
    end
    dag ~ model((λ=exp.(r[1,:]), μ=exp.(r[2,:]), η=η))
end

X, s = readdlm("example/9dicots-f01-25.csv", ',', Int, header=true)
tree = readnw(readline("example/9dicots.nw"))
dag, bound = CountDAG(X, s, tree)
rates = RatesModel(DLG(λ=ones(n), μ=ones(n), κ=0.0, η=0.66), fixed=(:κ,))
basemodel = PhyloBDP(rates, tree, bound)

model = branchrates(dag, basemodel)
chain = sample(model, NUTS(0.65), 500);

# ------------------------------------------------------------------------------
using Literate
Literate.markdown(
    joinpath(@__DIR__, "README.jl"),
    joinpath(@__DIR__, "../"),
    documenter=false, execute=false)
# using the execute-markdown branch now
