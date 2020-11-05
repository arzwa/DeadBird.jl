# Likelihood and other methods specific to the linear model
# Actually the matter is probably more about the branching property than about
# linearity, e.g. rates that grow quadratic with the population size might also
# admit a Csuros-Miklos like algorithm.  At any rate, only methods for the
# linear BDP are implemented here.

# NOTE: when having family-specific rates, we might get a performance boost
# when considering the node count bound for each family separately. Since
# the transition probabilities (W) in that case have to be computed for
# each family separately, it is not useful to compute values up to the 
# upper bound of the entire matrix for a node in the speices tree...

const ΛMTOL = 1e-6
const LMTOL = log(ΛMTOL)
approx1(x) = x ≈ one(x) ? one(x) : x
approx0(x) = x ≈ zero(x) ? zero(x) : x
probify(x) = max(min(x, one(x)), zero(x))

"""
    getϕψ(t, λ, μ)

[Not exported] Returns `ϕ = μ(eʳ - 1)/(λeʳ - μ)` where `r = t*(λ-μ)` and `ψ =
ϕ*λ/μ`, with special cases for λ ≈ μ. These methods should be implremented as
to prevent underflow/overflow issues.  Note these quantities are also called p
and q (in Csuros & Miklos) or α and β (in Bailey). Note that ϕ = P(Xₜ=0|X₀=1),
i.e. the extinction probability for a single particle.
"""
function getϕψ(t, λ, μ)
    if isapprox(λ, μ, atol=ΛMTOL)
        ϕ = probify(λ*t/(one(λ) + λ*t))
        return ϕ, ϕ
    else
        r = exp(t*(λ-μ))
        # for large values in the exponent, we get Inf, but
        # actually the true value is μ/λ. If we get -Inf, the 
        # true value should be 1.
        r ==  Inf && return μ/λ, one(μ)
        r == -Inf && return one(μ), λ/μ
        a = μ*(r-one(r))
        b = λ*r-μ
        ϕ = a/b
        return probify(ϕ), probify(ϕ*λ/μ)
    end
end

"""
    extp(t, λ, μ, ϵ)

[Not exported] Compute the extinction probability of a single lineage evolving
according to a linear BDP for time `t` with rate `λ` and `μ` and with
extinction probability of a single lineage at `t` equal to `ϵ`. This is
`∑ᵢℙ{Xₜ=i|X₀=1}ϵ^i`
"""
function extp(t, λ, μ, ϵ)
    # XXX: takes ϵ on probability scale!
    # NOTE: seems sufficiently stable that we don't need `probify`
    ϵ ≈ one(ϵ)  && return one(ϵ)
    ϵ ≈ zero(ϵ) && return getϕψ(t, λ, μ)[1]
    if isapprox(λ, μ, atol=ΛMTOL)
        e = one(ϵ) - ϵ
        return one(ϵ) - e/(μ*t*e + one(ϵ))
    else
        r = exp(t*(λ-μ))
        a = λ*r*(ϵ - one(ϵ))
        b = μ-λ*ϵ
        c = one(a) + a/b
        d = μ/λ
        return d + (one(d) - d)/c
    end
end

getϵ(n, i::Int) = n.data.ϵ[i]
setϵ!(n, i::Int, x) = n.data.ϵ[i] = x

"""
    getϕψ′(ϕ, ψ, ϵ)

[Not exported] Note that we take ϵ on a probability scale!
```
ϕ′ = [ϕ(1-ϵ) + (1-ψ)ϵ]/[1 - ψϵ]
ψ′ = [ψ(1-ϵ)]/[1-ψϵ]
```
Some edge cases are when ϵ is 1 or 0. Other edge cases may be relevant when ψ
and or ϕ is 1 or 0.
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

# NOTE that this does not involve the gain model!
function setϵ!(n::ModelNode{T}, rates::M) where {T,M<:LinearModel}
    isleaf(n) && return  # XXX or should we set ϵ to -Inf?
    θn = getθ(rates, n)
    if iswgd(n) || iswgt(n)
        c = first(children(n))
        ϵc = getϵ(c, 2)
        ϵn = iswgd(n) ? wgdϵ(θn.q, ϵc) : wgtϵ(θn.q, ϵc)
        setϵ!(c, 1, ϵn)
        setϵ!(n, 2, ϵn)
    else
        setϵ!(n, 2, zero(T))
        for c in children(n)
            θc = getθ(rates, c)
            ϵc = log(extp(distance(c), θc.λ, θc.μ, exp(getϵ(c, 2))))
            setϵ!(c, 1, ϵc)
            setϵ!(n, 2, getϵ(n, 2) + ϵc)
        end
    end
end

#wgdϵ(q, ϵ) = q*ϵ^2 + (one(q) - q)*ϵ
#wgtϵ(q, ϵ) = q*ϵ^3 + 2q*(one(q) - q)*ϵ^2 + (one(q) - q)^2*ϵ

# takes ϵ on log scale!
wgdϵ(q, ϵ) = logaddexp(log(q)+2ϵ, log1p(-q) + ϵ)
wgtϵ(q, ϵ) = logsumexp([log(q)+3ϵ, log(2q)+log1p(-q)+2ϵ, 2*log1p(-q)+ϵ])

# Conditional survival transition probability matrix
function setW!(n::ModelNode{T}, rates::V) where {T,V<:LinearModel}
    isroot(n) && return
    ϵ = getϵ(n, 2)
    θ = getθ(rates, n)
    if iswgdafter(n)
        wstar_wgd!(n.data.W, distance(n), θ, ϵ)
    elseif iswgtafter(n)
        wstar_wgt!(n.data.W, distance(n), θ, ϵ)
    else
        wstar!(n.data.W, distance(n), θ, ϵ)
    end
end

"""
    wstar!(w::Matrix, t, θ, ϵ)

Compute the transition probabilities for the conditional survival process
recursively (not implemented using recursion though!). Note that the resulting
transition matrix is *not* a stochastic matrix of some Markov chain.
"""
function wstar!(w::Matrix{T}, t, θ, ϵ) where T  # compute w* (Csuros Miklos '09)
    @unpack λ, μ, κ = θ
    l = size(w)[1]-1
    ϕ, ψ = getϕψ(t, λ, μ)  # p, q
    ϕ′, ψ′ = getϕψ′(ϕ, ψ, exp(ϵ))
    a = 1. - ϕ′
    b = 1. - ψ′
    c = log(a) + log(b)  # (1-p')*(1-q') in Csuros
    d = log(ψ′)  # q' in Csuros
    (κ/λ > zero(κ)) && (b > zero(b)) ? # gain model
        # if κ = λ this is a geometric pmf...
        w[1,:] = logpdf.(NegativeBinomial(κ/λ, b), 0:l) :
        w[1,1] = zero(T)
    for m=1:l, n=1:m
        w[n+1, m+1] = logaddexp(d + w[n+1, m], c + w[n, m])
    end
end

# XXX untested since brought on log-scale!
function wstar_wgd!(w, t, θ, ϵ)
    @unpack λ, μ, q = θ
    l1me = log1mexp(ϵ)
    a = log((1. - q) + 2q*exp(ϵ)) + l1me
    b = log(q) + 2l1me
    w[1,1] = zero(q)
    w[2,2] = a
    w[2,3] = b
    l = size(w)[1]-1
    for i=1:l, j=2:l
        w[i+1, j+1] = logaddexp(a + w[i,j], b + w[i, j-1]) 
    end
end

function wstar_wgt!(w, t, θ, ϵ)
    @unpack λ, μ, q = θ
    q1 = log(q)
    q2 = log(2q) + log(1. - q)
    q3 = 2log(1. - q)
    a = logsumexp([q1+l1me, log(2.)+q2+ϵ+l1me, log(3.)+q3+2ϵ+l1me])
    b = logaddexp(q2 + 2l1me, log(3.) + q3 + ϵ + 2l1me)
    c = q3 + 3l1me
    w[1,1] = zero(q)
    w[2,2] = a
    w[2,3] = b
    w[2,4] = c
    l = size(w)[1]-1
    for i=1:l, j=3:l
        w[i+1, j+1] = logsumexp([a+w[i, j], b+w[i, j-1], c+w[i, j-2]])
    end
end

# Root integration
function ∫root(p, rates, ϵ) 
    @unpack η = rates.params
    if rates.rootprior == :shifted 
        ℓ = ∫rootshiftgeometric(p, η, ϵ)
    elseif rates.rootprior == :geometric
        ℓ = ∫rootgeometric(p, η, ϵ)
    else
        throw("$(rates.rootprior) not implemented!")
    end
end

# We could work with types as well and use dispatch...
function conditionfactor(model)
    return if model.cond == :root
        nonextinctfromrootcondition(model)
    #elseif model.cond == :nowhere
    #    extinctnowherecondition(model)
    elseif model.cond == :none
        0.
    else
        throw("Condition not implemented! $(model.cond)")
    end
end

"""
    ∫rootshiftgeometric(ℓ, η, ϵ)

Integrate the loglikelihood at the root for the conditional process, with the
prior on the number of lineages existing (X) at the root a shifted geometric
distribution with mean 1/η, i.e. X ~ Geometric(η)+1

Σₙ ℓ[n] × Σᵢ (n+i, choose i) ϵⁱ(1-ϵ)ⁿ η(1-η)ⁿ⁺ⁱ⁻¹

`ℓ[i]` is ℙ{data|Y=i}, where Y is the number of lineages at the root that leave
observed descendants. `le` log extinction probablity lϵ.  This function
computes ℙ{data|X} based on ℙ{data|Y} (right?).  Assumes at least one ancestral
gene.
"""
@inline function ∫rootshiftgeometric(ℓ, η, lϵ)
    p = -Inf
    for i in 2:length(ℓ)
        f = (i-1)*log1mexp(lϵ) + (i-2)*log(1. - η) + log(η)
        f -= i*log1mexp(log(one(η) - η)+lϵ)
        p = logaddexp(p, ℓ[i] + f)
    end
    return p
end

"""
    ∫rootgeometric(ℓ, η, ϵ)

Σₙ ℓ[n] × Σᵢ (n+i, choose i) ϵⁱ(1-ϵ)ⁿ η(1-η)ⁿ⁺ⁱ

There are two differences with the shifted geometric formula, (1) exponent to
(1-η) is n+i instead of n+i-1, and (2) we start from the X=0 state, not the X=1
state
"""
@inline function ∫rootgeometric(ℓ, η, lϵ)
    p = -Inf
    for i in 1:length(ℓ)
        f = (i-1)*(log1mexp(lϵ) + log(1. - η)) + log(η)
        f -= i*log1mexp(log(one(η) - η)+lϵ)
        p = logaddexp(p, ℓ[i] + f)
    end
    return p
end

# This is the non-extinction in both clades stemming from the root condition
# XXX: assumes the shifted geometric prior!
# XXX: only for linear model, since it works from the extinction probability
# for a single lineage and assumes the branching property
function nonextinctfromrootcondition(model::LPhyloBDP)
    @unpack η = getθ(model.rates, root(model))
    lη = log(η)
    o  = root(model)
    ϵo = geomϵp(getϵ(o, 2), lη)  # XXX some ugly log(exp(log(exp...)))
    ϵc = map(c->geomϵp(getϵ(c, 1), lη), children(o)) |> logsumexp
    log(probify(1. - exp(ϵc) + exp(ϵo)))
end

geomϵp(lϵ, lη) = lη + lϵ -log1mexp(log1mexp(lη) + lϵ)

# XXX: see pgf technique (implemented in Whale) for nowhere extinct
# condition...


# Methods using the CountDAG data structure
"""
    loglikelihood!(dag::CountDAG, model::PhyloBDP)

Compute the log likelihood on the DAG using the Csuros & Miklos
algorithm.
"""
function loglikelihood!(dag::CountDAG, model::LPhyloBDP{T}) where T
    for level in dag.levels  # parallelism possible within levels
        Threads.@threads for n in level
            cm!(dag, n, model)
        end
    end
    ℓ = acclogpdf(dag, model) - dag.nfam*conditionfactor(model)
    #!isfinite(ℓ) && @warn "ℓ not finite"
    isfinite(ℓ) ? ℓ : -Inf
end
# NOTE: maybe a distributed approach using SharedArray or DArray would be more
# efficient, but it's much less convenient (and not so compatible with AD?)

function acclogpdf(dag::CountDAG, model::LPhyloBDP{T}) where T
    @unpack graph, ndata, parts = dag
    @unpack η = getθ(model.rates, root(model))
    ϵ = getϵ(root(model), 2)
    ℓ = zero(T)
    for n in outneighbors(graph, nv(graph))
        ℓ += ndata[n].count*∫root(parts[n], model.rates, ϵ)
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
    @unpack graph, ndata, parts = dag
    @unpack η = getθ(model.rates, model[1])
    ϵ = getϵ(model[1], 2)
    [∫root(parts[n], model.rates, ϵ) for n in nodes]
end

# Likelihood using the Profile(Matrix) data structure
# NOTE: condition is optional, because for a full matrix it is of course
# redundant to compute the same condition factor many times.  Nevertheless, we
# still want to have loglikelihood(profile) to give the correct loglikelihood
# value for a single profile as well.
function loglikelihood!(p::Profile,
        model::LPhyloBDP{T},
        condition=true) where T
    @unpack η = getθ(model.rates, root(model))
    for n in model.order
        cm!(p, n, model)
    end
    ℓ = ∫root(p.ℓ[1], model.rates, getϵ(root(model), 2))
    if condition
        ℓ -= conditionfactor(model)
    end
    isfinite(ℓ) ? ℓ : -Inf
end

# Major refactor
# Somwehat more intelligle, and doesn't allocate large matrices. However, it
# is not faster (but numerically more stable). It should be more amenable to
# further optimizations though...
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

"""
    logmatvecmul(A, x)

Compute log.(Ax) given log.(A) and log.(x).
Can this be optimized to have less allocs?  Doing it in-place didn't seem to
help much?
"""
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
            logaddexp(B[s+2,t], lϵ₁+B[s+1,t])
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
            @inbounds Aᵢ[n+1] = logaddexp(Aᵢ[n+1], lp)
        end
    end
    for n=0:k2  # this is 0 ... M[i]
        @inbounds A[n+1] = Aᵢ[n+1] - n*l1me
    end
end

