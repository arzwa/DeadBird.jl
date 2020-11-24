# Truncated matrix approaches for nonlinear birth-death process models with
# computations using a classical pruning algorithm.  The likelihood for DL, DLG
# and *WGD models can be computed using the CM algorithm but for more
# complicated models, we often need to truncate the state space and perform
# matrix exponentiation.  This is because the CM algorithm relies on the
# underlying process being a proper branching process, i.e. where evolution of
# each particle is independent of the other particles.  The `PhyloBDP` struct
# can be used directly for these purposes, we have the W field in the
# `ModelNode` object to store the transition probability matrices.  So we could
# just dispatch on the `RatesModel` to know whether we need the truncated state
# space models or whether we can use the CM algorithm. 

# HACK, a proper constructor would be nicer...
function nonlineardag(g::CountDAG, bound)
    newparts = map(g.parts) do x
        if x[end] == 0.
            y = fill(-Inf, bound+1)
            y[length(x)] = 0.
        else
            y = fill(NaN, bound+1)
        end
        y
    end
    CountDAG(g.graph, g.levels, g.ndata, newparts, g.nfam)
end

function setmodel!(model::PhyloBDP)
    @unpack order, rates = model
    for n in order
        setW!(n, rates)
    end
end

function setW!(n::ModelNode{T}, rates) where T
    isroot(n) && return
    Q = getQ(rates.params, n)
    n.data.W .= _exp(Q*distance(n))
end

# NOTE: for non-linear models the pgf formulation can't be used (because
# ℙ{extinction} = F(0) is only valid for processes with the branching property)
"""
Compute relevant extinction probabilities.
    A ≡ {extinction  left from root}
    B ≡ {extinction right from root}
    ℙ{¬extinct in left or right} = 1-ℙ{A ∪ B} = 1 - (ℙ{A} + ℙ{B} - ℙ{A ∩ B})
Note that ℙ{A} = ℙ{A,B} + ℙ{A,¬B} (marginalized)
"""
function nonextinctfromrootcondition(model::PhyloBDP{T}) where T
    @unpack rates, order = model
    @unpack m, Q, η = rates.params
    zerop = Profile(zeros(Int, length(order)), m+1, T)
    for n in order
        prune!(zerop, n, model)
    end
    o = root(model)
    # ℓϵroot probability of an extinct profile given the prior on the root
    ϵroot = exp(∫rootshiftgeometric(zerop.ℓ[id(o)], η))
    setϵ!(o, 2, exp(zerop.ℓ[id(o)][2]))  # this is correct
    ϵchildren = map(children(o)) do c
        𝑃 = c.data.W
        ϵ′ = 𝑃 * exp.(zerop.ℓ[id(c)])  # extinction Ps at beginning of edge
        ϵ = exp(∫rootshiftgeometric(log.(ϵ′), η))
        # this [π * 𝑃 * ℓ'] with π the geometric pdf (row vector)
        setϵ!(c, 1, ϵ′[2])  # this is correct
        ϵ
    end
    log(one(ϵroot) - ϵchildren[1] - ϵchildren[2] + ϵroot)
end

function loglikelihood!(dag::CountDAG, model)
    for level in dag.levels  # parallelism possible within levels
        # Threads.@threads for n in level
        for n in level
            prune!(dag, n, model)
        end
    end
    ℓ = acclogpdf(dag, model) - dag.nfam*conditionfactor(model)
    isfinite(ℓ) ? ℓ : -Inf
end

function acclogpdf(dag::CountDAG, model)
    @unpack graph, ndata, parts = dag
    @unpack η = getθ(model.rates, root(model))
    ϵ = log(probify(getϵ(root(model), 2)))
    ℓ = 0.
    for n in outneighbors(graph, nv(graph))
        ℓ += ndata[n].count*∫rootshiftgeometric(parts[n], η)
    end
    return ℓ
end

@inline function prune!(dag::CountDAG{T}, n, model) where T
    @unpack ndata, parts, graph = dag
    outdegree(graph, n) == 0 && return initpartsleaf!(dag, n, model)
    initparts!(dag, n, model)
    for c in outneighbors(graph, n)
        𝑃 = model[ndata[c].snode].data.W
        parts[n] .+= log.(𝑃 * exp.(parts[c]))
    end
end

function initpartsleaf!(dag::CountDAG{T}, n, model) where T
    @unpack ndata, parts = dag
    parts[n] = fill(-Inf, model.bound)
    parts[n][ndata[n].bound+1] = zero(T)
end

initparts!(dag::CountDAG{T}, n, model) where T =
    dag.parts[n] = zeros(T, model.bound)

# nonlinear models
function loglikelihood!(p::Profile, model, condition=true)
    @unpack η = getθ(model.rates, root(model))
    ϵ = log(probify(getϵ(root(model), 2)))
    for n in model.order
        prune!(p, n, model)
    end
    ℓ = ∫rootshiftgeometric(p.ℓ[1], η)
    if condition
        ℓ -= conditionfactor(model)
    end
    isfinite(ℓ) ? ℓ : -Inf
end

# Integrate a geometric distribution on the root for the nonlinear models.
# Here we do not use the conditional survival process,
# so we just need to multiply the likelihood values at the root
# by the corresponding prior probabilities (renormalized for finite support)
# and do the sum.
function ∫rootshiftgeometric(ℓ, η)
    n = length(ℓ)
    # pdf(Geometric) is very fast recursively implemented in Distributions.jl
    p = pdf.(Geometric(η), 1:n-1)
    p = log.(p /= sum(p))  # renormalize
    logsumexp(ℓ[2:end] .+ p)
end

@inline function prune!(p::Profile{T}, n, model) where T
    @unpack x, ℓ = p
    if isleaf(n)
        ℓ[id(n)][x[id(n)]+1] = 0.
        return
    end
    ℓ[id(n)] .= zero(T)
    for c in children(n)
        𝑃 = model[id(c)].data.W
        ℓ[id(n)] .+= log.(𝑃 * exp.(ℓ[id(c)]))
    end
end
