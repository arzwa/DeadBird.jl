import Random: AbstractRNG

const RootPrior = DiscreteUnivariateDistribution

@doc raw"""
    marginalize(prior, ℓvec, logϵ) 

Compute the log-likelihood of the data at the root by marginalizing the partial
conditional survival likelihoods at the root over the prior on the number of
genes at the root. This is the following sum

```math
\ell = \sum_{n=1}^{b} \ell[n] \Big( \sum_{i=0}^\infty 
    \binom{n+i}{i} \epsilon^i (1 - \epsilon)^n P\{X_o = n+i\} \Big)
```

Where `ℓ[n] = P{data|Yₒ=n}`, where `Yₒ` denotes the number of genes at the root
that leave observed descendants and `Xₒ` denotes the total number of genes at
the root, for which we specified the prior. `b` is the bound on the number of
surviving lineages, which is determined by the observed data. For many priors,
the innner infinite sum can be simplified to a closed form after some algebraic
manipulation and relying on the fact that `∑ binom(α + k - 1, k) z^k = (1 -
z)^(-k)` for `|z| < 1`.
"""
function marginalize end

@doc raw"""
    marginal_extinctionp(d, logϵ)

Compute the marginal log probability of having no observed descendants for a
branching process starting off with `n` genes given by a probability
distribution `d`  when the probability that a single gene goes extinct is `ϵ`.
This is:

```math
\sum_{k=1}^\infty ϵ^k P\{X₀ = k\}
```

For most priors a closed form can be obtained by manipulating the sum so that
it becomes a geometric series.
"""
function marginal_extinctionp end

# marginalize over geometric using closed form.
@inline function marginalize(p::Geometric, ℓvec, logϵ)
    ℓ = -Inf
    a = log1mexp(logϵ) + log(1 - p.p) 
    c = log(p.p)
    d = log1mexp(log(1 - p.p) + logϵ)
    for i in 1:length(ℓvec)
        f = (i-1)*a + c - i*d
        ℓ = logaddexp(ℓ, ℓvec[i] + f)
    end
    return ℓ
end

# marginalize over Poisson
@inline function marginalize(p::Poisson, ℓvec, logϵ)
    ℓ = -Inf
    r = p.λ*(1. - exp(logϵ))
    r < zero(r) && return ℓ
    d = Poisson(r) 
    for i in 1:length(ℓvec)
        ℓ = logaddexp(ℓ, ℓvec[i] + logpdf(d, i-1))
    end
    return ℓ
end

"""
    ShiftedGeometric

Geometric distribution with domain [1, 2, ..., ∞).
"""
struct ShiftedGeometric{T} <: RootPrior
    η::T
end

# perhaps better to store the Geometric distribution in the struct
Base.rand(rng::AbstractRNG, d::ShiftedGeometric) = rand(rng, Geometric(d.η)) + 1
Base.rand(rng::AbstractRNG, d::ShiftedGeometric, n::Int) = rand(rng, Geometric(d.η), n) .+ 1
Distributions.logpdf(d::ShiftedGeometric, k::Int) = logpdf(Geometric(d.η), k-1) 

# marginalize over ShiftedGeometric prior using closed form.
@inline function marginalize(p::ShiftedGeometric, ℓvec, logϵ)
    ℓ = -Inf
    a = log1mexp(logϵ)
    b = log(1 - p.η)
    c = log(p.η)
    d = log1mexp(b + logϵ)
    for i in 2:length(ℓvec)
        f = (i-1)*a + (i-2)*b + c - i*d
        ℓ = logaddexp(ℓ, ℓvec[i] + f)
    end
    return ℓ
end

# marginalize extinction probability
function marginal_extinctionp(p::ShiftedGeometric, logϵ)
    log(p.η) + logϵ - log1mexp(log(1 - p.η) + logϵ)
end

@doc raw"""
    ShiftedBetaGeometric(η, ζ)

Beta-Geometric compound distribution on the domain [1, 2, ..., ∞).  The pmf is
given by

```math
p_k = \frac{\mathrm{B}(\alpha + 1, \beta + k - 1)}{\mathrm{B}(\alpha, \beta)}
```

!!! note 
    We use the alternative parameterization using the mean `η = α/(α+β)` and
    *offset* 'sample size' `ζ = α + β - 1`, where `ζ > 0`. That is, we assume
    `α + β > 1`.
"""
struct ShiftedBetaGeometric{T} <: RootPrior
    η::T
    ζ::T
end

getαβ(d::ShiftedBetaGeometric) = d.η * (d.ζ + 1), (1 - d.η) * (d.ζ + 1)  

function Base.rand(rng::AbstractRNG, d::ShiftedBetaGeometric) 
    p = rand(Beta(getαβ(d)...))
    p = p <= zero(p) ? 1e-16 : p >= one(p) ? 1-1e-16 : p
    return rand(Geometric(p)) + 1
end

Base.rand(rng::AbstractRNG, d::ShiftedBetaGeometric, n::Int) = map(rand(rng, d), 1:n)

@doc raw"""
    marginalize(p::ShiftedBetaGeometric, ℓvec, logϵ)

If we do the series, and make use of the fact that `B(x,y+1) = B(x,y)*y/(x+y)`
We can obtain the following:

```math
\ell = \sum_{n=1}^b \ell[n] 
    \frac{\mathrm{B}(\alpha + 1, \beta + n + 1)}{\mathrm{B}(\alpha, \beta)}
    \frac{(1-\epsilon)^n}{(1-\epsilon \frac{\beta + n-1}{\alpha + \beta + n})^{n+1}}
```
"""
@inline function marginalize(p::ShiftedBetaGeometric, ℓvec, logϵ)
    α, β = getαβ(p)
    a = log1mexp(logϵ)
    b = logbeta(α + 1, β) + a - logbeta(α, β)  # base term 
    c = log(β) - log(α + β + 1) + a
    ℓ = -Inf
    for n in 2:length(ℓvec)
        f = b - n*log(1 - exp(logϵ) * (β + n - 2) / (α + β + n - 1))
        ℓ = logaddexp(ℓ, ℓvec[n] + f)
        b += c  # update term with Beta function terms
    end
    return ℓ
end

# marginalize extinction probability
function marginal_extinctionp(p::ShiftedBetaGeometric, logϵ)
    α, β = getαβ(p)
    p = logϵ + log(p.η) - log1mexp(logϵ + log(β) - log(β + α + 1))
    return p
end
