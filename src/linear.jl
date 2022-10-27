# Likelihood and other methods specific to the linear model

# We might get speedups by implementing custom frules for ChainRules, and get
# them to work with ForwardDiff...

# NOTE: when having family-specific rates, we might get a performance boost
# when considering the node count bound for each family separately. Since the
# transition probabilities (W) in that case have to be computed for each family
# separately, it is not useful to compute values up to the upper bound of the
# entire matrix for a node in the species tree...

# Not sure how to choose this constant, fairly important for cancellation
# errors though
const ΛMTOL = 1e-6  
const LMTOL = log(ΛMTOL)

approx1(x) = x ≈ one(x) ? one(x) : x
approx0(x) = x ≈ zero(x) ? zero(x) : x
probify(x) = max(min(x, one(x)), zero(x))

function _logaddexp(a, b)
    (a == -Inf && b == -Inf) && return -Inf
    a == -Inf && return b
    b == -Inf && return a
    return logaddexp(a, b)
end

# Conditional survival transition probabilities and extinction probabilities
# ==========================================================================
"""
    getϕψ(t, λ, μ)

Returns `ϕ = μ(eʳ - 1)/(λeʳ - μ)` where `r = t*(λ-μ)` and `ψ = ϕ*λ/μ`, with
special cases for λ ≈ μ. These methods should be implemented as to prevent
underflow/overflow issues.  Note these quantities are also called p and q (in
Csuros & Miklos) or α and β (in Bailey). Note that ϕ = P(Xₜ=0|X₀=1), i.e. the
extinction probability for a single gene.
"""
function getϕψ(t, λ, μ)
    if λ == zero(λ)  # gain+loss 
        ϕ = probify(1. - exp(-μ*t))
        return ϕ, zero(λ)
    elseif isapprox(λ, μ, atol=ΛMTOL)  # (approximately) critical case
        ϕ = probify(λ*t/(one(λ) + λ*t))
        return ϕ, ϕ
    else
        r = exp(t*(λ-μ))
        # if r ~ Inf, then the extinction probability should be 0, and ψ as well
        r == Inf && return zero(r), zero(r)
        # if r ~ 0 (exponent -> -Inf), we get almost sure extinction
        r ≈ zero(r) && return one(μ), λ/μ
        a = μ*(r-one(r))
        b = λ*r-μ
        ϕ = a/b
        return probify(ϕ), probify(ϕ*λ/μ)
    end
end

"""
    extp(t, λ, μ, ϵ)

Compute the extinction probability of a single lineage evolving according to a
linear BDP for time `t` with rate `λ` and `μ` and with extinction probability
of a single lineage at `t` equal to `ϵ`. This is `∑ᵢℙ{Xₜ=i|X₀=1}ϵ^i`

!!! note
    Takes ϵ on a [0,1] scale
"""
function extp(t, λ, μ, ϵ)
    # NOTE: seems sufficiently stable that we don't need `probify`
    ϵ ≈ one(ϵ)  && return one(ϵ)
    ϵ ≈ zero(ϵ) && return getϕψ(t, λ, μ)[1]
    if isapprox(λ, μ, atol=ΛMTOL)
        e = one(ϵ) - ϵ
        return one(ϵ) - e/(μ*t*e + one(ϵ))
    elseif λ ≈ zero(λ)  # gain+loss model
        return (1.0 - exp(-μ*t)) + ϵ * exp(-μ*t)
    else
        r = exp(t*(λ-μ))
        a = λ*r*(ϵ - one(ϵ))
        b = μ-λ*ϵ
        c = one(a) + a/b
        d = μ/λ
        return d + (one(d) - d)/c
    end
end

@doc raw"""
    getϕψ′(ϕ, ψ, ϵ)

Adjusted ϕ and ψ for a linear BDP process with extinction probability ϵ after
the process.

```math
\phi' = \frac{\psi(1-\epsilon) + (1 - \psi) \epsilon}{1 - \psi \epsilon}

\psi' = \frac{\psi(1-\epsilon)}{1 - \psi \epsilon}
```

Some edge cases are when ϵ is 1 or 0. Other edge cases may be relevant when ψ
and or ϕ is 1 or 0.

!!! note
    We take ϵ on [0,1] scale.
"""
function getϕψ′(ϕ, ψ, ϵ)
    ϵ ≈ one(ϵ)  && return one(ϕ), zero(ψ)
    ϵ ≈ zero(ϵ) && return ϕ, ψ
    c = one(ψ) - ψ*ϵ
    a = one(ϵ) - ϵ
    ϕ′ = (ϕ*a + (one(ψ)-ψ)ϵ)/c
    ψ′ = ψ*(one(ϵ)-ϵ)/c
    any([ϕ′, ψ′] .> one(ϵ))  && @debug "Problem!" ϕ ψ ϵ ϕ′ ψ′
    any([ϕ′, ψ′] .< zero(ϵ)) && @debug "Problem!" ϕ ψ ϵ ϕ′ ψ′
    probify(ϕ′), probify(ψ′)
end

"""
    setϵ!(n, rates)

Set the extinction probabilities for node `n` given the rates.
"""
function setϵ!(n::ModelNode{T}, rates) where T
    θ = getθ(rates, n)
    if isleaf(n) 
        ϵ2 = T(-Inf)
    elseif iswgm(n)
        ϵ2 = wgmϵ(θ.q, getk(n), getϵ(n[1], 1))
    else
        ϵ2 = sum(getϵ.(children(n), 1))
    end
    setϵ!(n, 2, ϵ2)
    ϵ1 = log(extp(distance(n), θ.λ, θ.μ, exp(ϵ2)))
    setϵ!(n, 1, ϵ1)
end

"""
    wgmϵ(q, k, logϵ)

Compute the log-extinction probability of a single lineage going through a
k-plication event, given the extinction probability of a single lineage after
the WGM event. Assumes the single parameter WGM retention model (assuming a
single gene before the WGM, the number of retained genes after the WGM is a rv
X' = 1 + Y where Y is Binomial(k - 1, q)).
"""
wgmϵ(q, k, logϵ) = logϵ + (k - 1) * log(q * (exp(logϵ) - 1.) + 1.)

"""
    setW!(n, rates)

Compute the conditional survival transition probability matrix for teh branch
leading to node `n`.
"""
function setW!(n::ModelNode{T}, rates) where T
    isroot(n) && return
    if iswgmafter(n)
        wstar_wgm!(n, rates)
    else
        wstar!(n.data.W, distance(n), getθ(rates, n), getϵ(n, 2))
    end
end

"""
    wstar!(W::Matrix, t, θ, ϵ)

Compute the transition probabilities for the conditional survival process
recursively (not implemented using recursion though!). Note that the resulting
transition matrix is *not* a stochastic matrix of some Markov chain.
"""
function wstar!(W::AbstractMatrix{T}, t, θ, ϵ) where T
    @unpack λ, μ, κ = θ
    l = size(W, 1) - 1
    ϕ , ψ  = getϕψ(t, λ, μ)        # p , q  in Csuros
    ϕ′, ψ′ = getϕψ′(ϕ, ψ, exp(ϵ))  # p', q' in Csuros
    a = 1. - ϕ′
    b = 1. - ψ′
    c = log(a) + log(b) 
    d = log(ψ′)
    if κ == 0.
        # no gain (κ = 0.)
        #W[1,:] .= T(-80.)  # XXX
        #W[:,1] .= T(-80.)  # XXX ForwardDiff issues!
        W[1,1] = zero(T)
    else
        r = κ/λ
        if (zero(r) < r < 1e10) && (b > zero(b))
            # gain+duplication+loss, NegativeBinomial contribution from gain
            # if λ = κ this is a geometric distribution
            # if λ is so small as to dwarfed by κ this is a Poisson distribution
            # we use the gain-loss model in the latter case
            W[1,:] = logpdf.(NegativeBinomial(r, b), 0:l)
        elseif r >= 1e10  
            # gain+loss (ignore duplication if λ ≈ 0), Poisson contribution from gain
            r′ = κ * ϕ * (1. - exp(ϵ)) / μ  # recall ϕ = (1-e^(-μt))
            W[1,:] = logpdf.(Poisson(r′), 0:l)     
        end
    end
    for m=1:l, n=1:m   # should we go from 1 to m?
        W[n+1, m+1] = _logaddexp(d + W[n+1, m], c + W[n, m])
    end
end

"""
    wstar_wgm!

Compute transition probabilities for conditional survival process across a WGM
node. This computes the entries for the W matrix of a `wgmafter` node. The event
is a `k`-ploidization and we assume a model where a duplicated gene is retained
post-WGM independently with probability `q`.

!!! note
    Transition probabilities are different when we model excess number of genes
    in a family compared to total family sizes.
"""
function wstar_wgm!(n, rates)
    θ = getθ(rates, n)
    ϵ = getϵ(n, 2)
    k = getk(parent(n))
    if is_excessmodel(rates)
        wstar_wgm_ne_nonrecursive!(n.data.W, k, θ, ϵ)
    else
        wstar_wgm!(n.data.W, k, θ, ϵ )
    end
end

function wstar_wgm!(W, k, θ, logϵ)
    @unpack λ, μ, q = θ
    l = size(W, 1) - 1
    W[1,1] = zero(λ)  # gain is not involved at wgm nodes 0->0 w.p. 1
    ϵ = exp(logϵ)
    for j=1:k
        p = zero(λ)
        for n=j:k
            p_ = binomial(k-1, n-1) * q^(n-1) * (1. - q)^(k-n)  
            # n-1 out of k-1 excess genes retained, n genes in total
            p += p_ * binomial(n, j) * (1. - ϵ)^j * ϵ^(n-j)     
            # j of n retained genes survive
        end
        W[2, j+1] = log(p)
    end
    for i=2:l, j=i:(min(k*i,l))
        # XXX issues with ForwardDiff -Inf doesn't work!
        W[i+1, j+1] = -500. #-Inf  # for safety 
        # for a `k`-plication, we have min(j-1, k) terms to sum to get pᵢⱼ
        for n=1:min(j-i+1,k)
            W[i+1, j+1] = _logaddexp(W[i+1, j+1], W[2, n+1] + W[i, j-n+1])
        end
    end
end

@doc raw"""
    wstar_wgm_ne!(W, k, θ, logϵ)

When we use the linear BDIP on the number of excess genes, the calculation for
the conditional survival transition matrix is slightly different! The full
solution seems to be

```math
w_{ij} = \sum_{n=0}^a 
    \binomial{i+n}{j}(1-\epsilon)^j \epsilon^{n-j} 
    \binomial{a}{n} q^n (1-q)^{a-n}
```

where `a = (k-1)*(i+1)`.

!!! warn
    This leads to overflows in binomial, we should figure out a recursion...
"""
function wstar_wgm_ne_nonrecursive!(W, k, θ, logϵ)
    @unpack q = θ
    l = size(W, 1) - 1
    l1me = log1mexp(logϵ) 
    imax = (l+1)*k
    if q ≈ zero(q)
        W[1,1] = zero(q)
        for i=1:l
            @inbounds W[i+1, i+1] = W[i,i] + l1me
        end
        return
    end
    B = bctable(imax, imax)  
    for i=0:l, j=i:min(l, k*(i+1)-1)
        a = (k-1)*(i+1)
        W[i+1, j+1] = -Inf
        for n=0:a
            #p_ = binomial(i+n, j) * (1-ϵ)^j * ϵ^(i+n-j)
            #p_ = B[i+n+1, j+1] * (1-ϵ)^j * ϵ^(i+n-j)
            p_ = log(B[i+n+1, j+1]) + j*l1me + logϵ * (i+n-j) 
            p_ += log(B[a+1, n+1]) + n*log(q) + (a-n) * log(1. - q)
            #wij += p_ * binomial(a, n) * q^n * (1-q)^(a-n)
            #wij += p_ * B[a+1, n+1] * q^n * (1-q)^(a-n)
            W[i+1, j+1] = _logaddexp(W[i+1, j+1], p_)
        end
    end
end

@memoize function bctable(imax, jmax)
    X = zeros(Int, imax+1, jmax+1)
    X[1:end,1] .= 1
    for i=1:imax, j=1:i
        X[i+1, j+1] = X[i, j] + X[i, j+1]
    end
    return X
end

# haven't figure this out yet...
function wstar_wgm_ne!(W, k, θ, logϵ)
    @unpack q = θ
    l = size(W, 1) - 1
    ϵ = exp(logϵ)
    for j=0:l
        a = k-1
        wij = zero(ϵ) 
        for n=0:a
            p_ = binomial(n, j) * (1-ϵ)^j * ϵ^(n-j)
            wij += p_ * binomial(a, n) * q^n * (1-q)^(a-n)
        end
        W[1, j+1] = log(wij)
    end
    nϵ = log1mexp(logϵ)
    for i=1:l, j=i:(min(k*(i+1)-1,l))
        W[i+1, j+1] = -Inf  # for safety 
        for n=0:min(j-i,k-1)
            W[i+1, j+1] = _logaddexp(W[i+1, j+1], W[1, n+1] + W[i, j-n] + nϵ)
        end
    end
end

"""
    setw!(W, θ, t)

Compute transition probability matrix for the ordinary (not conditional on
survival that is) birth-death process. Using the recursive formulation of
Csuros & Miklos.
"""
function setw!(W::AbstractMatrix{T}, θ, t) where T
    @unpack λ, μ, κ = θ
    l = size(W, 1) - 1
    ϕ, ψ = getϕψ(t, λ, μ)  # p , q  in Csuros
    lϕ = log(ϕ)
    lψ = log(ψ)
    a = 1. - ϕ
    b = 1. - ψ
    c = log(a) + log(b) 
    d = log(1 - ϕ - ψ)
    r = κ/λ
    if (zero(r) < r < 1e10) && (b > zero(b))
        # dup+gain+loss 
        W[1,:] = logpdf.(NegativeBinomial(r, b), 0:l)
    elseif r >= 1e10  
        # gain+loss (ignore duplication if λ ≈ 0), Poisson contribution from gain
        ξ = κ*(1-exp(-μ*t))/μ
        W[1,:] = logpdf.(Poisson(ξ), 0:l)     
    else
        # no gain (κ = 0.)
        W[1,1] = zero(T)
    end
    for n=1:l
        W[n+1, 1] = lϕ + W[n, 1]
        W[n+1, 2] = _logaddexp(lϕ + W[n, 2], c + W[n, 1])
        for m=2:l
            x = _logaddexp(lψ + W[n+1, m], d + W[n, m])
            W[n+1, m+1] = _logaddexp(x, lϕ + W[n, m+1])
        end
    end
end

# ordinary transition probability matrix (not conditional on survival) for WGM
function setw_wgm!(W, k, θ)
    @unpack λ, μ, q = θ
    l = size(W, 1) - 1
    W[1,1] = zero(λ)  # gain is not involved at wgm nodes 0->0 w.p. 1
    for j=1:k  # initialization
        p = zero(λ)
        for n=j:k
            p += binomial(k-1, n-1) * q^(n-1) * (1. - q)^(k-n)  
            # n-1 out of k-1 excess genes retained, n genes in total
        end
        W[2, j+1] = log(p)
    end
    for i=2:l, j=i:(min(k*i,l))
        W[i+1, j+1] = -Inf  # for safety 
        # for a `k`-plication, we have min(j-1, k) terms to sum to get pᵢⱼ
        for n=1:min(j-i+1,k)
            W[i+1, j+1] = _logaddexp(W[i+1, j+1], W[2, n+1] + W[i, j-n+1])
        end
    end
end


# Conditioning factors
# ====================
"""
    conditionfactor(model)

Compute the condition factor for the model for the associated data filtering
strategy. 
"""
function conditionfactor(model)
    return if model.cond == :root
        nonextinctfromrootcondition(model)
    elseif model.cond == :observed  # XXX not well-tested
        marginal_extinctionp(model.rootp, getϵ(root(model), 2))
    elseif model.cond == :all
        conditionfactor(model, model.rootp)
    elseif model.cond == :none
        0.
    else
        throw("Condition not implemented! $(model.cond)")
    end
end

"""
    nonextinctfromrootcondition(model)

Compute the probability that a family existing at the root of the species tree
leaves observed descendants in both clades stemming from the root, i.e. does
not go extinct in any of the two clades stemming from the root. This uses the
marginalization of the extinction probability over the prior distribution on
the number of genes at the root using [`marginal_extinctionp`](@ref)
"""
function nonextinctfromrootcondition(model)
    o  = root(model)
    ϵo = marginal_extinctionp(model.rootp, getϵ(o, 2)) 
    ϵc = -Inf
    for c in children(o)
        ϵc = _logaddexp(ϵc, marginal_extinctionp(model.rootp, getϵ(c, 1)))
    end
    log(probify(1. - exp(ϵc) + exp(ϵo)))
end


# Loglikelihood methods using the CountDAG data structure
# =======================================================
"""
    loglikelihood!(dag::CountDAG, model::PhyloBDP)

Compute the log likelihood on the DAG using the Csuros & Miklos algorithm.
"""
function loglikelihood!(dag::CountDAG, model::LPhyloBDP{T}) where T
    for level in dag.levels  # parallelism possible within levels
        Threads.@threads for n in level
            cm!(dag, n, model)
        end
    end
    ℓ = acclogpdf(dag, model) - dag.nfam*conditionfactor(model)
    isfinite(ℓ) ? ℓ : -Inf
end

function acclogpdf(dag::CountDAG, model::LPhyloBDP{T}) where T
    @unpack graph, ndata, parts = dag
    ϵ = getϵ(root(model), 2)
    ℓ = zero(T)
    for n in outneighbors(graph, nv(graph))
        ℓ += ndata[n].count * marginalize(model.rootp, parts[n], ϵ)
    end
    return ℓ
end

# Mixture model, note that every site is `mixed` independently, we can't just
# sum the full data likelihood for each component!
function loglikelihood!(dag::CountDAG,
        model::MixtureModel{VF,VS,<:LPhyloBDP{T}}) where {VF,VS,T}
    @unpack graph, ndata = dag
    K = length(model.components)
    nodes = outneighbors(graph, nv(graph))
    # a matrix to store for each site pattern the likelihood for each component
    matrix = zeros(T, length(nodes), K)
    for i=1:K
        m = model.components[i]
        p = model.prior.p[i]
        for level in dag.levels  # parallelism possible within levels
            Threads.@threads for n in level
                cm!(dag, n, m)
            end
        end
        matrix[:,i] .= sitepatterns_ℓ(dag, m, nodes) .+ 
            log(p) .- conditionfactor(m) 
        # NOTE: the condition factor differs for the different components,
        # and we apply it for each site pattern
    end
    ℓs = vec(logsumexp(matrix, dims=2))
    ℓ = sum([ndata[n].count*ℓs[i] for (i,n) in enumerate(nodes)])
    isfinite(ℓ) ? ℓ : -Inf
end


# computes the site loglikelihood for each site pattern
function sitepatterns_ℓ(dag, model, nodes)
    ϵ = getϵ(model[1], 2)
    [marginalize(model.rootp, dag.parts[n], ϵ) for n in nodes]
end

# compute posterior probabilities for a single profile in a mixture model
function componentps(model::MixtureModel, y::Profile)
    K = length(model.components)
    l = map(1:K) do i
        m = model.components[i]
        p = model.prior.p[i]
        logpdf(m, y) + log(p)
    end
    exp.(l .- logsumexp(l))
end

function componentps(model::MixtureModel, y::ProfileMatrix)
    mapreduce(i->componentps(model, y[i]), hcat, 1:size(y,1)) |> permutedims
end

# Likelihood using the Profile(Matrix) data structure 
# ===================================================

# NOTE: condition is optional, because for a full matrix it is of course
# redundant to compute the same condition factor many times.  Nevertheless, we
# still want to have loglikelihood(profile) to give the correct loglikelihood
# value for a single profile as well.
function loglikelihood!(p::Profile,
        model::LPhyloBDP{T},
        condition=true) where T
    for n in model.order
        cm!(p, n, model)
    end
    ℓ = marginalize(model.rootp, p.ℓ[1], getϵ(root(model), 2))
    if condition
        ℓ -= conditionfactor(model)
    end
    return isfinite(ℓ) ? ℓ : -Inf
end


# Csuros & Miklos algorithm for computing the conditional survival likelihood
# ===========================================================================
"""
    cm!(dag, node, model)

Compute the conditional survival probabilities at `n` using the Csuros & Miklos
(2009) algorithm.  This assumes the `model` already contains the computed
transition probability matrices `W` and that the partial loglikelihood vectors
for the child nodes in the DAG are already computed and available.
"""
@inline function cm!(dag::CountDAG{T}, n, model) where T
    # n is a graph node (Int)
    @unpack graph, ndata, parts = dag
    if !isassigned(parts, n)
        parts[n] = fill(T(-Inf), ndata[n].bound+1)
    end
    if outdegree(graph, n) == 0  # leaf case
        parts[n][end] = zero(T) 
        return
    end
    cm_inner_dag!(dag, n, model, T)
end

@inline function cm!(profile::Profile{T}, n, model) where T
    # n is a node from the model
    @unpack x, ℓ = profile
    bound = length(ℓ[id(n)])
    if isleaf(n)
        ℓ[id(n)][end] = zero(T)
        return
    end
    cm_inner_profile!(ℓ, x, n, model, T)
end

# Some code repitition for DAG/profile, but trying to merge lead to efficiency 
# losses for profile...
@inline function cm_inner_dag!(dag, n, model, ::Type{T}) where T
    @unpack parts, ndata, graph = dag
    kcum = 0
    ϵcum = zero(T)
    for (i, ki) in enumerate(outneighbors(graph, n))
        kid   = model[ndata[ki].snode]  # child node in model
        kmaxi = ndata[ki].bound 
        @unpack W = kid.data
        extpi = getϵ(kid, 1)
        kcum′ = kcum + kmaxi
        ϵcum′ = ϵcum + extpi
        B = cm_getB(T, W, parts[ki], kcum, kcum′, kmaxi, extpi)
        i == 1 ? 
            cm_getA1!(parts[n], B, kcum′, ϵcum′, extpi) :
            cm_getAi!(parts[n], B, kcum, kcum′, ϵcum, ϵcum′, kmaxi, extpi)
        kcum = kcum′
        ϵcum = ϵcum′
    end
end

@inline function cm_inner_profile!(ℓ, x, n, model, ::Type{T}) where T
    kcum = 0
    ϵcum = zero(T)
    for (i,ki) in enumerate(children(n))
        @unpack W = model[id(ki)].data
        kmaxi = x[id(ki)]
        extpi = getϵ(ki, 1)
        kcum′ = kcum + kmaxi
        ϵcum′ = ϵcum + extpi
        B = cm_getB(T, W, ℓ[id(ki)], kcum, kcum′, kmaxi, extpi)
        i == 1 ? 
            cm_getA1!(ℓ[id(n)], B, kcum′, ϵcum′, extpi) :
            cm_getAi!(ℓ[id(n)], B, kcum, kcum′, ϵcum, ϵcum′, kmaxi, extpi)
        kcum = kcum′
        ϵcum = ϵcum′
    end
end

# Compute log.(Ax) given log.(A) and log.(x).
function logmatvecmul(A::AbstractMatrix{T}, x::Vector{T}) where T
    y = similar(x)
    for i=1:length(x)
        @inbounds y[i] = logsumexp(A[i,:] .+ x)
    end
    return y
end

# inplace version...
function _logmatvecmul!(B, A::AbstractMatrix{T}, x::Vector{T}) where T
    for i=1:length(x)
        @inbounds B[i,1] = logsumexp(A[i,:] .+ x)
    end
end

# @inbounds matters quite a lot to performance! but beware when 
# changing stuff...
@inline function cm_getB(T, W, L, k1, k2, mi, lϵ₁)
    B = fill(T(-Inf), (k2-k1+1, k2+1))
    #@views @inbounds B[:,1] = logmatvecmul(W[1:mi+1,1:mi+1], L)
    @views @inbounds _logmatvecmul!(B, W[1:mi+1,1:mi+1], L)
    for t=1:k1, s=0:mi  # this is 0...M[i-1] & 0...mi
        @inbounds Bts = s == mi ?
            B[s+1,t] + lϵ₁ : 
            _logaddexp(B[s+2,t], lϵ₁+B[s+1,t])
        @inbounds B[s+1,t+1] = Bts
    end
    return B
end

@inline function cm_getA1!(A, B, k2, ϵ2, lϵ₁)
    l1me = log1mexp(ϵ2)
    for n=0:k2  # this is 0 ... M[i]
        @inbounds A[n+1] = B[n+1,1] - n*l1me
    end
end

@inline function cm_getAi!(A::Vector{T}, B, k1, k2, ϵ1, ϵ2, mi, lϵ₁) where T
    #@info "cm_getAi" k1 k2 ϵ1 ϵ2 mi
    Aᵢ = fill(T(-Inf), k2+1)
    p = exp(ϵ1)
    l1me = log1mexp(ϵ2)
    for n=0:k2 
        tmax = min(k1, n)
        tmin = max(n-mi, 0)
        for t=tmin:tmax
            s = n-t
            # n, t and s correspond to 0-based indices (true counts)
            @inbounds lp = binomlogpdf(n, p, s) + A[t+1] + B[s+1,t+1]
            @inbounds Aᵢ[n+1] = _logaddexp(Aᵢ[n+1], lp)
        end
    end
    for n=0:k2  # this is 0 ... M[i]
        @inbounds A[n+1] = Aᵢ[n+1] - n*l1me
    end
end

