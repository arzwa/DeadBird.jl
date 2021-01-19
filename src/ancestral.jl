# Ancestral state reconstruction from conditional survival likelihoods
# The core functionality here is obtaining the posterior distribution for
# gene counts at ancestral nodes.

# NOTE: the pmf has to be computed up to an upper bound, but sampling could be
# done exact if we take the trouble I guess?
#
# NOTE: currently only for ProfileMatrix
#
# NOTE: should assess correctness using simulations
"""
    AncestralSampler(model, bound)

A wrapper that contains the transition probability matrices for the transient
distributions for the PhyloBDP (not conditioned on survival) along each branch.
"""
struct AncestralSampler{T,M}
    W::Vector{Matrix{T}}  # transition probability matrices
    model::M  
end

function AncestralSampler(model, bound)
    W = [fill(-Inf, bound+1, bound+1) for i=1:length(model)]
    for n in model.order[1:end-1]
        θ = getθ(model.rates, n)
        t = distance(n)
        setw!(W[id(n)], θ, t) 
    end
    return AncestralSampler(W, model)
end

getbound(spl::AncestralSampler) = size(spl.W, 1) - 1

"""
    Transient

Transient distribution P(X(t)|X(0)=k). This is a simple struct for the sampler.
"""
struct Transient{T}
    p::Vector{T}
end

Transient(spl::AncestralSampler, node, X) = Transient(spl.W[id(node)][X+1,:])
Distributions.logpdf(d::Transient, X) = d.p[X+1]

# normalize log-probabilities
function plognormalize(logp)
    logp[.!isfinite.(logp)] .= -Inf
    x = exp.(sort(logp .- maximum(logp), rev=true))
    return x ./ sum(x)
end

"""
    sample_ancestral(spl::AncestralSampler, x)

Sample a set of ancestral states using a pre-order traversal over the tree.
This assumes the partial likelihoods are available in `x`.
"""
sample_ancestral(spl::AncestralSampler, x) = sample_ancestral(Random.default_rng(), spl, x)

function sample_ancestral(rng::AbstractRNG, spl::AncestralSampler, X::ProfileMatrix)
    X = mapreduce(i->sample_ancestral(rng, spl, X[i]), hcat, 1:size(X, 1))
    permutedims(X)
end

function sample_ancestral(rng::AbstractRNG, spl::AncestralSampler, x::Profile)
    profile = copy(x.x)
    bound = getbound(spl)
    function walk(node, X)
        isleaf(node) && return
        prior = isroot(node) ? rootprior(spl.model) : Transient(spl, node, X)
        X = sample_ancestral_node(rng, node, x, prior, bound) 
        profile[id(node)] = X
        for c in children(node)
            walk(c, X)
        end
    end
    walk(getroot(spl.model), nothing)
    return profile
end

"""
    sample_ancestral_node(rng, node, x, prior, bound)

Sample ancestral state for node `node` with `prior` prior and relevant partial
likelihoods computed in `x`. The prior refers to either a root prior
distribution or the transient probability distribution of the process given the
parent state.
"""
function sample_ancestral_node(rng::AbstractRNG, node, x, prior, bound)
    p = ppmf(node, x, prior, bound)
    rand(rng, Categorical(p))
end

function sample_ancestral_node(rng::AbstractRNG, node, x, prior, bound, n)
    p = ppmf(node, x, prior, bound)
    rand(rng, Categorical(p), n)
end

"""
    ppmf(node, x, prior, bound)

Compute the posterior pmf for the ancestral state at node `node`, where `x`
holds the partial likelihoods somewhere, and `prior` corresponds to the root or
transition probability density that acts as a prior on the node state.
"""
ppmf(n::ModelNode, x::Profile, d, b) = ppmf(x.ℓ[id(n)], getϵ(n, 2), d, b)
ppmf(n::ModelNode, x, d, b) =  ppmf(x, getϵ(n, 2), d, b)

function ppmf(ℓvec, logϵ, prior, bound)
    l1mϵ = log1mexp(logϵ)
    p = fill(-Inf, bound)
    for n=1:bound
        for k=0:min(n, length(ℓvec)-1)
            p_nk = ℓvec[k+1] + (n - k)*logϵ + k*l1mϵ + logpdf(prior, n)
            p[n] = logaddexp(p[n], p_nk)
        end
    end
    return plognormalize(p)
end


## For the DAG, this is not correct...
#function sample_ancestral(rng::AbstractRNG, spl::AncestralSampler, dag::CountDAG)
#    @unpack graph, levels, ndata, parts, nfam = dag
#    @unpack model = spl
#    profile = Matrix{Int64}(undef, nfam, length(spl.model))
#    bound = getbound(spl)
#    function walk(node, Xs, idx)
#        # Xs are the parent X's
#        snode = model[ndata[node].snode]
#        if isleaf(snode)
#            profile[idx, id(snode)] .= ndata[node].bound
#            return
#        end
#        ℓvec = parts[node]
#        Xs = if isroot(snode) 
#            prior = rootprior(spl.model) 
#            sample_ancestral_node(rng, snode, ℓvec, prior, bound, ndata[node].count)
#        else
#            map(X->sample_ancestral_node(rng, snode, ℓvec, Transient(spl, snode, X), bound), Xs)
#        end
#        profile[idx, id(snode)] .= Xs
#        i = first(idx)
#        for c in outneighbors(graph, node)
#            cidx = i:(i+ndata[c].count-1)
#            walk(c, Xs, cidx)
#            i = last(cidx) + 1
#        end
#    end
#    i = 1
#    for n in dag.levels[end]
#        idx = i:(i+ndata[n].count-1) 
#        @info "" idx
#        walk(n, nothing, idx)
#        i = last(idx) + 1
#    end
#    profile
#end
