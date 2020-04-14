# We need a data structure that summarizes the entire data set. Or find any other way to arrange the computations more economically. Any unique subtree should have its likelihood computed onnly once. I would prefer to do this in a two stage kind of way by (1) first summarizing the count data with the species tree in a data structure that captures the redundancy across families and (2) an algorithm that than operates on that data structure.  Alternatively we could store computations on the go, a bit like a custom memoization, but not sure whether that would be efficient, I'd rather know exactly from beforehand the amount of `cm!` calls we will make. The structure I was having a feeling about turns out to be a DAG (no suprise I guess).

# I think if I got this + the CM algorithm nicely working + AD, it would be worth examining in more detail how it would compare to CAFE (and perhaps Count)?  [NOTE: CAFE uses Nelder-Mead apparently, while Count uses L-BFGS]. Should compare ML application with CAFE both in terms of speed and accuracy (a bit a shame Csuros & Miklos didn't compare accuracy of the conditional survival approach with a pruning algorithm approach?).

# NOTE: any such approach will be more tricky to implement the rjMCMC algrithm from Beluga for... We would have to localize *all* nodes in the graph where a new parent got inserted and update the order...

# NOTE: distributed computing combined with AD will be more challenging I guess, since it is not directly obvious what to parallelise... I guess computations at each 'level' of the species tree could in principle be parallelized, but to get AD to work with that seems challenging to say the least. However, maybe AD does work with threads or so?

# NOTE: mixtures that are not marginalized are another tricky thing for this approach. In general we lose flexibility whenever the model is no longer iid over families, or more precisely when the sampling probability computed in the main likelihood routine no longer involves the assumption of iidness across the entire data set. If we would be able to take subgraphs corresponding to parts of the data efficiently, this approach might still enable speed-ups compared to computing partial likelihoods independently for each family.

# Also NOTE that the same approch could work in ordinary phylogenetics (under a fixed topology), could be interesting, as the resulting reduction in computational burden might be more substantial there (not sure if this is however not already taken care of in PAML, BEAGLE, ...)

# The basic approach using this dat structure to represent the gene counts is
# (1) Obtain the DAG
# (2) Obtain a postordering
# (3) Compute in order, store the partial likelihoods
# (4) Obtain total likelihood

# NOTE, we need to keep a map from each node to the species tree node as well
# to be able to fetch parameters in the non-constant rates cases... Maybe first
# focus on the constant rates case though...

using LightGraphs, StatsBase, Parameters, StatsFuns

# I think a representation like the following is good
struct NodeData{I}
    snode::I
    count::Int
    bound::Int
end

struct CountDAG{G,I,T}
    graph::SimpleDiGraph{G}  # the DAG, with vertices ordered in a post-order
    ndata::Vector{NodeData{I}}
    parts::Vector{Vector{T}}
end

Base.show(io::IO, dag::CountDAG) = write(io, "CountDAG($(dag.graph))")

# The partial likelihood vectors were pulled out of the `NodeData`, because the type will cause issues when doing AD I guess.

# An alternative implementation would be to directly implement a DAGNode type with parents, children, the partial likelihood vector etc.

# NOTE: this is recursive, makes use of zip to combine clades
function builddag(matrix::Matrix, names, tree)
    colindex = Dict(s=>i for (i,s) in enumerate(names))
    dag = SimpleDiGraph()
    ndata = NodeData{typeof(id(tree))}[]
    parts = Vector{Float64}[]
    function walk(n)
        if isleaf(n)
            x = matrix[:,colindex[name(n)]]
            y = add_leaves!(dag, ndata, parts, x, id(n))
        else
            x = zip([walk(c) for c in children(n)]...)
            y = add_internal!(dag, ndata, parts, x, id(n))
            isroot(n) && add_root!(dag, ndata, y, id(n))
        end
        return y
    end
    walk(tree)
    CountDAG(dag, ndata, parts)
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
    [idmap[xᵢ] for xᵢ in x]
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
    [idmap[xᵢ] for xᵢ in x]
end

function add_root!(dag, ndata, x, n)
    add_vertex!(dag); i = nv(dag)
    for j in unique(x) add_edge!(dag, i, j) end
end

function likelihood!(dag, model)
    for n in 1:nv(dag.graph)-1
        outdegree(dag.graph, n) == 0 && continue
        cm!(dag, n, model)
    end
    ℓ = acclogpdf(dag, model) - condition(model)
    isfinite(ℓ) ? ℓ : -Inf
end

function acclogpdf(dag, model)
    @unpack graph, ndata, parts = dag
    η = model.rates.η
    ϵ = log(getϵ(model[1], 2))
    ℓ = 0.
    for n in outneighbors(graph, nv(graph))
        ℓ += ndata[n].count*∫rootgeom(parts[n], η, ϵ)
    end
    return ℓ
end

"""
    ∫rootgeom(ℓ, η, ϵ)

Integrate the loglikelihood at the root according to a shifted
geometric prior with mean 1/η and log extinction probablity ϵ.
Assumes at least one ancestral gene.
"""
@inline function ∫rootgeom(ℓ, η, lϵ)
    p = -Inf
    for i in 2:length(ℓ)
        f = (i-1)*log1mexp(lϵ) + log(η) + (i-2)*log(1. - η)
        f -= i*log1mexp(log(1. - η)+lϵ)
        p = logaddexp(p, ℓ[i] + f)
    end
    return p
end

function condition(model)
    lη = log(model.rates.η)
    cf = zero(lη)
    for c in children(model[1])
        ϵ = geomϵp(log(getϵ(c, 1)), lη)
        if ϵ > zero(lη)
            @warn "Invalid probability at `condition`, returning -Inf" ϵ
            return -Inf
        end
        cf += log1mexp(ϵ)
    end
    return cf
end

geomϵp(lϵ, lη) = lη + lϵ -log1mexp(log1mexp(lη) + lϵ)

@inline function cm!(dag, n, model)
    @unpack graph, ndata, parts = dag
    outdegree(graph, n) == 0 && return
    dnode = ndata[n]
    mnode = model[dnode.snode]
    kids = outneighbors(graph, n)
    kmax = [ndata[k].bound for k in kids]
    kcum = cumsum([0 ; kmax])
    ϵcum = cumprod([1.; [getϵ(c, 1) for c in children(mnode)]])
    B = fill(-Inf, (dnode.bound+1, kcum[end]+1, length(kmax)))
    A = fill(-Inf, (kcum[end]+1, length(kmax)))
    for (i, kid) in enumerate(kids)
        child = model[ndata[kid].snode]
        mi = kmax[i]
        Lk = parts[kid]
        bk = length(Lk)
        # Wc = child.data.W[1:dnode.bound+1, 1:dnode.bound+1]
        # B[:, 1, i] = log.(Wc * exp.(Lc[1:dnode.bound+1]))
        Wc = child.data.W[1:bk, 1:bk]
        B[1:bk, 1, i] = log.(Wc * exp.(Lk))
        ϵ₁ = log(getϵ(child, 1))
        for t=1:kcum[i], s=0:mi  # this is 0...M[i-1] & 0...mi
            B[s+1,t+1,i] = s == mi ?
                B[s+1,t,i] + ϵ₁ : logaddexp(B[s+2,t,i], ϵ₁+B[s+1,t,i])
        end
        if i == 1
            l1me = log(one(ϵ₁) - ϵcum[2])
            for n=0:kcum[i+1]  # this is 0 ... M[i]
                A[n+1,i] = B[n+1,1,i] - n*l1me
            end
        else
            # XXX is this loop as efficient as it could? I guess not...
            p = probify(ϵcum[i])
            for n=0:kcum[i+1], t=0:kcum[i]
                s = n-t
                (s < 0 || s > mi) && continue
                lp = binomlogpdf(n, p, s) + A[t+1,i-1] + B[s+1,t+1,i]
                A[n+1,i] = logaddexp(A[n+1,i], lp)
            end
            l1me = log(one(ϵ₁) - ϵcum[i+1])
            for n=0:kcum[i+1]  # this is 0 ... M[i]
                A[n+1,i] -= n*l1me
            end
        end
        parts[n] = A[:,end]
    end
end

# Now check AD


# Some tests for the DAG builder
using DelimitedFiles, NewickTree, Test, BenchmarkTools

X, s = readdlm("example/9dicots-f01-100.csv", ',', Int, header=true)
tree = readnw(readline("example/9dicots.nw"))
dag = builddag(X, s, tree)
g = dag.graph
@test outdegree(g, nv(g)) == length(unique(eachrow(X)))
@test sum([dag.ndata[i].count for i in outneighbors(g, nv(g))]) == size(X)[1]
# The graph contains 360 nodes, so I guess 360 calculations later. Naive calculation would result in 100 × 17 calculations I guess, and calculation of unique rows 81 × 17 = 1377. So we could expect some speed up

X, s = readdlm("example/9dicots-f01-1000.csv", ',', Int, header=true)
tree = readnw(readline("example/9dicots.nw"))
dag = builddag(X, s, tree)
g = dag.graph
@test outdegree(g, nv(g)) == length(unique(eachrow(X)))
@test sum([dag.ndata[i].count for i in outneighbors(g, nv(g))]) == size(X)[1]
# here we would go from 660 * 17 = 11220 to 1606

X, s = readdlm("example/9dicots-f01-25.csv", ',', Int, header=true)
tree = readnw(readline("example/9dicots.nw"))
dag = builddag(X, s, tree)
g = dag.graph
@test outdegree(g, nv(g)) == length(unique(eachrow(X)))
@test sum([dag.ndata[i].count for i in outneighbors(g, nv(g))]) == size(X)[1]

mmax = maximum([n.bound for n in dag.ndata])
r = ConstantDLG(λ=1.0, μ=1.2, κ=0.0 , η=0.9)
m = PhyloBDP(r, tree, 24+1)
@show likelihood!(dag, m)
@btime likelihood!(dag, m)

for n in outneighbors(dag.graph, nv(dag.graph))
    @show dag.ndata[n].count, round.(dag.parts[n][1:4], digits=2)
end

# previous version (25 gene families, 9 dicots)
# julia> @btime logpdf!(model, data)
#   1.953 ms (7648 allocations: 1.31 MiB)
# -251.0358357105553

# (1000 gene families, 9 dicots)
# julia> @btime logpdf!(model, data)
#   85.427 ms (312904 allocations: 58.62 MiB)
# -12724.094725213423
