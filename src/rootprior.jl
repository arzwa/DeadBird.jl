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

For many priors a closed form can be obtained by manipulating the sum so that
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
    α::T
    β::T
    ShiftedBetaGeometric(η::T, ζ::T) where T = new{T}(η, ζ, getαβ(η, ζ)...)
end

ShiftedBetaGeometric(η, ζ) = ShiftedBetaGeometric(promote(η, ζ)...)

getαβ(η, ζ) = (α=η * (ζ + 1), β=(1 - η) * (ζ + 1))
logp(α, β, k) = logbeta(α + 1, β + k) - logbeta(α, β)

function Base.rand(rng::AbstractRNG, d::ShiftedBetaGeometric) 
    p = rand(Beta(d.α, d.β))
    p = p <= zero(p) ? 1e-16 : p >= one(p) ? 1-1e-16 : p
    return rand(Geometric(p)) + 1
end

Base.rand(rng::AbstractRNG, d::ShiftedBetaGeometric, n::Int) = map(i->rand(rng, d), 1:n)
Distributions.logpdf(d::ShiftedBetaGeometric, k) = logp(d.α, d.β, k - 1)

"""
    marginalize(p::ShiftedBetaGeometric, ℓvec, logϵ, imax=100)

There seems to be no closed form for this, but we can devise a recursion and
obtain a near-exact solution efficiently.
"""
@inline function marginalize(p::ShiftedBetaGeometric, ℓvec, logϵ, imax=20)
    @unpack η, α, β = p
    a = log1mexp(logϵ)
    A = log(η)  # A10 term
    ℓ = -Inf
    for k in 2:length(ℓvec)
        f = _innerseries_shiftedbg(A, logϵ, k - 1, α, β, imax)
        ℓ = logaddexp(ℓ, ℓvec[k] + (k - 1) * a + f)
        # recursion on k, update to get the *next* term
        A += log(β + k - 2) - log(α + β + k - 1) 
    end
    return ℓ
end

# Inner series for marginalizing over a shifted beta-geometric distribution
# This uses a recursion on `i` to compute the terms in the partial sum
function _innerseries_shiftedbg(A, logϵ, k, α, β, imax)
    S = A  # partial sum
    for i=1:imax
        a = logϵ + log(k + i) + log(β + k + i - 2)  # numerator
        b = log(i) + log(α + β + k + i - 1)  # denominator
        A = A + a - b  # A_{k,i} term
        S = logaddexp(S, A)  # add term to series
    end
    return S 
end

# marginalize extinction probability (no closed form AFAIK, but efficient recursion)
function marginal_extinctionp(p::ShiftedBetaGeometric, logϵ, kmax=20)
    @unpack η, α, β = p
    A = log(η) + logϵ  # first term
    p = A  # partial sum
    for k=2:kmax
        A += logϵ + log(β + k - 2) - log(α + β + k - 1)  # next term Ak
        p = logaddexp(p, A)  # add term to series
    end
    return p
end

"""
    BetaGeometric(η, ζ)
"""
struct BetaGeometric{T} <: RootPrior
    η::T
    ζ::T
    α::T
    β::T
    BetaGeometric(η::T, ζ::T) where T = new{T}(η, ζ, getαβ(η, ζ)...)
end

BetaGeometric(η, ζ) = BetaGeometric(promote(η, ζ)...)

function Base.rand(rng::AbstractRNG, d::BetaGeometric) 
    p = rand(Beta(d.α, d.β))
    p = p <= zero(p) ? 1e-16 : p >= one(p) ? 1-1e-16 : p
    return rand(Geometric(p))
end

Base.rand(rng::AbstractRNG, d::BetaGeometric, n::Int) = map(rand(rng, d), 1:n)
Distributions.logpdf(d::BetaGeometric, k) = logp(d.α, d.β, k)

"""
   loglikelihood(d::BetaGeometric, ks::Vector{Int})

Loglikelihod for a vector of counts `ks`, i.e. `[x1, x2, x3, ...]` where `x1`
is the number of times k=1 is observed in the data, `x2` the number of times
k=2 is observed, etc.
"""
function Distributions.loglikelihood(d::BetaGeometric, ks::Vector{Int})
    logp = 0.
    for (k, count) in enumerate(ks)
        logp += count * logpdf(d, k-1)
    end
    return logp
end

"""
    marginalize(p::ShiftedBetaGeometric, ℓvec, logϵ, imax=100)

There seems to be no closed form for this, but we can devise a recursion,
analogous to the ShiftedBetaGeometric case. This could probably share code, but
I'll have it separate for now.
"""
@inline function marginalize(p::BetaGeometric, ℓvec, logϵ, imax=20)
    @unpack η, α, β = p
    a = log1mexp(logϵ)
    A = log(η)  # base term 
    ℓ = -Inf
    for k in 1:length(ℓvec)
        f = _innerseries_bg(A, logϵ, k - 1, α, β, imax)
        ℓ = logaddexp(ℓ, ℓvec[k] + (k - 1) * a + f)
        # recursion on k, update to get the *next* term
        A += log(β + k - 1) - log(α + β + k) 
    end
    return ℓ
end

# This could probably be combined with _innerseries_shiftedbg but perhaps
# clearer to keep separate
function _innerseries_bg(A, logϵ, k, α, β, imax)
    S = A  # partial sum
    for i=1:imax
        a = logϵ + log(k + i) + log(β + k + i - 1)  # numerator
        b = log(i) + log(α + β + k + i)  # denominator
        A = A + a - b  # A_{k,i} term
        S = logaddexp(S, A)  # add term to series
    end
    return S 
end

