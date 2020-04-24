# I think a representation like the following is good, it does not have to be  modified at any point during agorithms
struct NodeData{I}
    snode::I
    count::Int
    bound::Int
end

"""
    CountDAG{T,G,I}

The directed acyclic graph (DAG) representation of a phylogenetic
profile for an (assumed known) species tree. This is a [multitree](
https://en.wikipedia.org/wiki/Multitree)

    CountDAG(matrix, header, tree::Node)

Build the gene count DAG from `matrix` with a `header` corresponding
to leaf names of `tree`. Returns the bound as well (for the PhyloBDP
model constructor).
"""
struct CountDAG{T,G,I}  # I'd prefer this to have one type parameter fewer
    graph ::SimpleDiGraph{G}  # the DAG, with vertices ordered in a post-order
    levels::Vector{Vector{G}}
    ndata ::Vector{NodeData{I}}
    parts ::Vector{Vector{T}}
    nfam  ::Int
end

Base.show(io::IO, dag::CountDAG) = write(io, "CountDAG($(dag.graph))")

# An alternative implementation would be to directly implement a DAGNode type with parents, children, the partial likelihood vector etc.

# The copy function is important for AD applications. It's quite cheap when using `similar`.
copydag(g, T) = CountDAG(g.graph, g.levels, g.ndata,
    similar(g.parts,Vector{T}), g.nfam)

# constructor, returns the bound as well (for the PhyloBDP model constructor)
function CountDAG(matrix::Matrix, names, tree)
    colindex = Dict(s=>i for (i,s) in enumerate(names))
    dag = SimpleDiGraph()
    ndata = NodeData{typeof(id(tree))}[]
    parts = Vector{Float64}[]
    levels = Dict{Int,Vector{Int}}()
    function walk(n, l)
        if isleaf(n)
            x = matrix[:,colindex[name(n)]]
            y = add_leaves!(dag, ndata, parts, x, id(n))
        else
            x = zip([walk(c, l+1) for c in children(n)]...)
            y = add_internal!(dag, ndata, parts, x, id(n))
            isroot(n) && add_root!(dag, ndata, y, id(n))
        end
        haskey(levels, l) ? union!(levels[l], unique(y)) : levels[l] = unique(y)
        return y
    end
    walk(tree, 1)
    bound = maximum([n.bound for n in ndata])+1
    levels = collect(values(sort(levels, rev=true)))
    (dag=CountDAG(dag, levels, ndata, parts, size(matrix)[1]),
        bound=bound)
end

# We might want to implement a dedicated DAG builder for single gene families, in a way that we can use the same functions for computing the likelihood etc. when we do not want to lump all the data together (because of different model parameters for different families).

function add_leaves!(dag, ndata, parts, x, n)  # x are leaf counts
    idmap = Dict()
    for (k,v) in countmap(x)
        push!(ndata, NodeData(n, v, k))
        push!(parts, [fill(-Inf, k); 0.])
        add_vertex!(dag)
        idmap[k] = nv(dag)
    end
    [idmap[xᵢ] for xᵢ in x]  # returns for each leaf count the corresponding node that was added to the graph
end

function add_internal!(dag, ndata, parts, x, n)  # x are tuples of DAG nodes
    idmap = Dict()
    for (k,v) in countmap(collect(x))
        bound = sum([ndata[i].bound for i in k])
        push!(ndata, NodeData(n, v, bound))
        push!(parts, fill(-Inf, bound+1))
        add_vertex!(dag); i = nv(dag)
        for j in k add_edge!(dag, i, j) end
        idmap[k] = i
    end
    [idmap[xᵢ] for xᵢ in x]  # returns for each split the corresponding node that was added to the graph
end

function add_root!(dag, ndata, x, n)
    add_vertex!(dag); i = nv(dag)
    for j in unique(x) add_edge!(dag, i, j) end
end

Distributions.logpdf(m::PhyloBDP{T}, x::CountDAG) where T =
    loglikelihood!(copydag(x, T), m)

"""
    loglikelihood!(dag::CountDAG, model::PhyloBDP)

Compute the log likelihood on the DAG using the Csuros & Miklos
algorithm.
"""
function loglikelihood!(dag, model)
    # for n in 1:nv(dag.graph)-1 cm!(dag, n, model) end
    for level in dag.levels  # parallelism possible within levels
        Threads.@threads for n in level
            cm!(dag, n, model)
        end
    end
    ℓ = acclogpdf(dag, model) - dag.nfam*conditionfactor(model)
    isfinite(ℓ) ? ℓ : -Inf
end
# NOTE: I guess a distributed approach using SharedArray or DArray could be more efficient, but it's much less convenient (and not so compatible with AD?)

function acclogpdf(dag, model)
    @unpack graph, ndata, parts = dag
    @unpack η = getθ(model.rates, root(model))
    ϵ = log(probify(getϵ(root(model), 2)))
    ℓ = 0.
    for n in outneighbors(graph, nv(graph))
        ℓ += ndata[n].count*∫rootgeometric(parts[n], η, ϵ)
    end
    return ℓ
end

# implementation for a arginalized mixture model
function loglikelihood!(dag, model::PhyloBDP{T,V}) where
        {T,V<:RatesModel{T,<:GammaMixture}}
    @unpack params, rrates = model.rates.params
    @unpack graph, ndata = dag
    nodes = outneighbors(graph, nv(graph))
    K = length(rrates)
    matrix = zeros(T, length(nodes),K)
    for (i,rr) in enumerate(rrates)
        setmodel!(model.order, params*rr)
        for level in dag.levels  # parallelism possible within levels
            Threads.@threads for n in level
                cm!(dag, n, model)
            end
        end
        matrix[:,i] .= sitepatterns_ℓ(dag, model, nodes)
    end
    ℓs = vec(logsumexp(matrix, dims=2)) .- log(K)
    ℓ = sum([ndata[n].count*ℓs[i] for (i,n) in enumerate(nodes)]) -
        dag.nfam*rootcondition(model)
    isfinite(ℓ) ? ℓ : -Inf
end

function sitepatterns_ℓ(dag, model, nodes)
    @unpack graph, ndata, parts = dag
    @unpack η = getθ(model.rates, model[1])
    ϵ = log(probify(getϵ(model[1], 2)))
    [∫rootgeometric(parts[n], η, ϵ) for n in nodes]
end

"""
    cm!(dag, node, model)

Compute the conditional survival probabilities at `n` using
Csuros & Miklos algorithm. This assumes the `model` already contains
the computed transition probability matrices `W` and that the partial
loglikelihood vectors for the child nodes in the DAG are already
computed and available.
"""
@inline function cm!(dag::CountDAG{T}, n, model) where T
    @unpack graph, ndata, parts = dag
    if outdegree(graph, n) == 0  # leaf case
        isassigned(parts, n) && return
        parts[n] = [fill(T(-Inf), ndata[n].bound) ; zero(T)]
        return
    end
    dnode = ndata[n]
    mnode = model[dnode.snode]
    kids = outneighbors(graph, n)
    kmax = [ndata[k].bound for k in kids]
    kcum = cumsum([0 ; kmax])
    keps = [getϵ(c, 1) for c in children(mnode)]
    ϵcum = cumprod([1.; keps])
    B = fill(T(-Inf), (dnode.bound+1, kcum[end]+1, length(kmax)))
    A = fill(T(-Inf), (kcum[end]+1, length(kmax)))
    for (i, kid) in enumerate(kids)
        @unpack W = model[ndata[kid].snode].data
        cm_inner!(i, A, B, W, parts[kid],
            ϵcum, kcum, kmax[i], log(keps[i]))
    end
    parts[n] = A[:,end]
end

# this can and should be shared with a non-DAG implementation
@inline function cm_inner!(i, A, B, W, L, ϵcum, kcum, mi, lϵ₁)
    @inbounds B[1:mi+1, 1, i] = log.(W[1:mi+1, 1:mi+1] * exp.(L))
    for t=1:kcum[i], s=0:mi  # this is 0...M[i-1] & 0...mi
        @inbounds B[s+1,t+1,i] = s == mi ?
            B[s+1,t,i] + lϵ₁ : logaddexp(B[s+2,t,i], lϵ₁+B[s+1,t,i])
    end
    if i == 1
        l1me = log(probify(one(lϵ₁) - ϵcum[2]))
        for n=0:kcum[i+1]  # this is 0 ... M[i]
            @inbounds A[n+1,i] = B[n+1,1,i] - n*l1me
        end
    else
        # XXX is this loop as efficient as it could? I guess not...
        p = probify(ϵcum[i])
        for n=0:kcum[i+1], t=0:kcum[i]
            s = n-t
            (s < 0 || s > mi) && continue
            @inbounds lp = binomlogpdf(n, p, s) + A[t+1,i-1] + B[s+1,t+1,i]
            @inbounds A[n+1,i] = logaddexp(A[n+1,i], lp)
        end
        l1me = log(probify(one(lϵ₁) - ϵcum[i+1]))
        for n=0:kcum[i+1]  # this is 0 ... M[i]
            @inbounds A[n+1,i] -= n*l1me
        end
    end
end

# ## Notes
# We need a data structure that summarizes the entire data set. Or find any other way to arrange the computations more economically. Any unique subtree should have its likelihood computed only once. I would prefer to do this in a two stage kind of way by (1) first summarizing the count data with the species tree in a data structure that captures the redundancy across families after which (2) we use an algorithm that than operates on that data structure.  Alternatively we could store computations on the go, a bit like a custom memoization, but that would probably not be very efficient. Also, I'd rather know exactly from beforehand the amount of `cm!` calls we will make.

# It seems the data structure we were looking for is a DAG. Apparently (and unsurprisingly) this approach to combine data (sites) along subtrees is already known in phylogenetics (in Yang's textbook it is referred to as 'partial site patterns'). I'm unsure however whether the explicit treatment of the problem as a DAG has been pointed out before. Also I'm unsure whether this is implemented in any high-performance phylogenetics library.

# It would be worth examining in more detail how the approach here compares to CAFE (and perhaps Count)? Note that CAFE uses Nelder-Mead apparently, while Count uses L-BFGS (but not sure how they compute gradients). A comparison with CAFE for ML estimation both in terms of speed and accuracy would be interesting (a bit a shame Csuros & Miklos didn't compare accuracy of the conditional survival approach with a pruning algorithm approach?).

# NOTE: the DAG approach will be more tricky to implement the rjMCMC algrithm from Beluga for... We would have to localize *all* nodes in the graph where a new parent got inserted and update the order...

# NOTE: distributed computing combined with AD will be more challenging I guess, since it is not directly obvious what to parallelise... I guess computations at each 'level' of the species tree could in principle be parallelized, but to get AD to work with that seems challenging to say the least. However, maybe AD does work out of the box with threads?

# NOTE: mixtures that are not marginalized are another tricky thing for this approach. In general we lose flexibility whenever the model is no longer iid over families, or more precisely when the sampling probability computed in the main likelihood routine no longer involves the assumption of iidness across the entire data set. If we would be able to take subgraphs corresponding to parts of the data efficiently, this approach might still enable speed-ups compared to computing partial likelihoods independently for each family.
