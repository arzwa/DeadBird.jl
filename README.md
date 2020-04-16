BirdDad

This is a reimplementation of the core functionalities previousy implemented in the `Beluga` library.

## Maximum-likelihood estimation

```julia
using BirdDad, Optim, NewickTree, DelimitedFiles
```

Read in the data and tree

```julia
X, s = readdlm("example/9dicots-f01-1000.csv", ',', Int, header=true)
tree = readnw(readline("example/9dicots.nw"))
```

Construct the data object

```julia
dag, bound = CountDAG(X, s, tree)
```

Construct the model

```julia
rates = RatesModel(
    ConstantDLG(λ=1.0, μ=1.2, κ=0.0, η=1/mean(X)),
    fixed=(:η, :κ))
model = PhyloBDP(rates, tree, bound)

f, ∇f = mle_problem(dag, model)
@time out = optimize(f, ∇f, randn(2), BFGS())
t = transform(model.rates.trans, out.minimizer)
```

Note that this is an order of magnitude faster than the `R` implementation in WGDgc (where I measured a run time ≈35s for the same data and initial conditions)

## Side note:
The Hessian of the negative loglikelihood evluated at the MLE is equal to the observed Fisher information $I(\hat{\theta})$. The estimated standard errors for the ML estimates $\mathrm{SE}(\hat{\theta}) = 1/\sqrt{I(\hat{\theta})}$ can be obtained as follows

```julia
using ForwardDiff, LinearAlgebra

function ff(x::Vector{T}) where T
    rates = model.rates((λ=x[1], μ=x[2]))
    m = PhyloBDP(rates, tree, bound)
    d = copydag(dag, T)
    -loglikelihood!(d, m)
end
1 ./ diag(.√(ForwardDiff.hessian(ff, [t.λ, t.μ])))
```

Note that, sadly, we cannot use `ForwardDiff.hessian(f, out.minimizer)` to obtain this, since that would calculate the hessian in transformed space.

## Bayesian inference using DynamicHMC

```julia
using DynamicHMC, LogDensityProblems, Random, DynamicHMC.Diagnostics
```

First, we need to code our 'LogDensityProblem' struct with an associated
function that returns the log posterior density. Note that the problem should
a subtype of `BirdDad.Problem`

```julia
struct ConstantDLGProblem <: BirdDad.Problem
    model
    data
end

function (p::ConstantDLGProblem)(θ)
    @unpack λ, μ, η, κ = θ
    ℓ = loglikelihood(p, θ)
    π = logpdf(MvLogNormal(ones(2)), [λ, μ]) +
         logpdf(Exponential(0.1), κ) +
         logpdf(Beta(1,3), η)
    return ℓ + π
end

r = RatesModel(ConstantDLG(λ=1.0, μ=1.2, κ=0.01 , η=1/mean(X)))
m = PhyloBDP(r, tree, bound)
p = ConstantDLGProblem(m, dag)
t = trans(p)
P = TransformedLogDensity(t, p)
∇P = ADgradient(:ForwardDiff, P);
results = mcmc_with_warmup(Random.GLOBAL_RNG, ∇P, 100);
posterior = transform.(t, results.chain)
@info summarize_tree_statistics(results.tree_statistics)
```

------------------------------------------------------------------------------

```julia
using Literate
Literate.markdown(
    joinpath(@__DIR__, "README.jl"),
    joinpath(@__DIR__, "../"),
    documenter=false, execute=false)
```

using the execute-markdown branch now

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

