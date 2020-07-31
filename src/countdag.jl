# I think a representation like the following is good,
# it does not have to be  modified at any point during agorithms
struct NodeData{I}
    snode::I   # species tree node
    count::Int
    bound::Int
end

"""
    CountDAG{T,G,I}

The directed acyclic graph (DAG) representation of a phylogenetic profile
for an (assumed known) species tree.
This is a [multitree](https://en.wikipedia.org/wiki/Multitree)

    CountDAG(matrix, header, tree::Node)

Build the gene count DAG from `matrix` with a `header`
corresponding to leaf names of `tree`.
Returns the bound as well (for the PhyloBDP model constructor).
"""
struct CountDAG{T,G,I}  # I'd prefer this to have one type parameter fewer
    graph ::SimpleDiGraph{G}  # the DAG, with vertices ordered in a post-order
    levels::Vector{Vector{G}}
    ndata ::Vector{NodeData{I}}
    parts ::Vector{Vector{T}}
    nfam  ::Int
end

Base.show(io::IO, dag::CountDAG) = write(io, "CountDAG($(dag.graph))")

# An alternative implementation would be to directly implement a DAGNode type
# with parents, children, the partial likelihood vector etc.

# The copy function is important for AD applications.
# It's quite cheap when using `similar`.
copydag(g, T) = CountDAG(g.graph, g.levels, g.ndata,
    similar(g.parts, Vector{T}), g.nfam)

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

# constructor, returns the bound as well (for the PhyloBDP model constructor)
CountDAG(df, tree) = CountDAG(Matrix(df), names(df), tree)
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
    bound = maximum([n.bound for n in ndata])
    levels = collect(values(sort(levels, rev=true)))
    cdag = CountDAG(dag, levels, ndata, parts, size(matrix)[1])
    (dag=cdag, bound=bound)
end

# We might want to implement a dedicated DAG builder for single gene families,
# in a way that we can use the same functions for computing the likelihood etc.
# when we do not want to lump all the data together
# (because of different model parameters for different families).
# NOTE: well we have the profile matrix now, but still
# it could be interesting to be able to efficiently obtain 'subgraphs'
# from the CountDAG.

"""
    add_leaves!(dag, ndata, parts, x, n)

For a species tree leaf node `n`,
this adds the vector of (gene) counts `x`
for that species to the graph.
This returns for each gene family
the corresponding node that was added to the graph
"""
function add_leaves!(dag, ndata, parts, x, n)
    idmap = Dict()
    for (k,v) in countmap(x)
        push!(ndata, NodeData(n, v, k))
        push!(parts, [fill(-Inf, k); 0.])
        add_vertex!(dag)
        idmap[k] = nv(dag)
    end
    [idmap[xᵢ] for xᵢ in x]
end

"""
    add_internal!(dag, ndata, parts, x, n)

For a species tree internal node `n`,
this adds the gene family nodes associated with `n` to the graph
and provides the bound on the number of lineages that survive
to the present below `n` for each gene family.
Note that `x` is a vector of tuples of DAG nodes
that each will be joined into a newly added node.
The resulting nodes are returned.

NOTE: I believe this also works for multifurcating species trees
(like the Csuros Miklos algorithm does too)
"""
function add_internal!(dag, ndata, parts, x, n)
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

Distributions.logpdf(m::PhyloBDP{T}, x::CountDAG) where T =
    loglikelihood!(copydag(x, T), m)

Distributions.logpdf(m::MixtureModel{VF,VS,<:PhyloBDP{T}},
    x::CountDAG) where {VF,VS,T} = loglikelihood!(copydag(x, T), m)


# ## Notes
# We need a data structure that summarizes the entire data set. Or find any other way to arrange the computations more economically. Any unique subtree should have its likelihood computed only once. I would prefer to do this in a two stage kind of way by (1) first summarizing the count data with the species tree in a data structure that captures the redundancy across families after which (2) we use an algorithm that than operates on that data structure.  Alternatively we could store computations on the go, a bit like a custom memoization, but that would probably not be very efficient. Also, I'd rather know exactly from beforehand the amount of `cm!` calls we will make.

# It seems the data structure we were looking for is a DAG. Apparently (and unsurprisingly) this approach to combine data (sites) along subtrees is already known in phylogenetics (in Yang's textbook it is referred to as 'partial site patterns'). I'm unsure however whether the explicit treatment of the problem as a DAG has been pointed out before. Also I'm unsure whether this is implemented in any high-performance phylogenetics library.

# It would be worth examining in more detail how the approach here compares to CAFE (and perhaps Count)? Note that CAFE uses Nelder-Mead apparently, while Count uses L-BFGS (but not sure how they compute gradients). A comparison with CAFE for ML estimation both in terms of speed and accuracy would be interesting (a bit a shame Csuros & Miklos didn't compare accuracy of the conditional survival approach with a pruning algorithm approach?).

# NOTE: the DAG approach will be more tricky to implement the rjMCMC algrithm from Beluga for... We would have to localize *all* nodes in the graph where a new parent got inserted and update the order...

# NOTE: distributed computing combined with AD will be more challenging I guess, since it is not directly obvious what to parallelise... I guess computations at each 'level' of the species tree could in principle be parallelized, but to get AD to work with that seems challenging to say the least. However, maybe AD does work out of the box with threads?

# NOTE: mixtures that are not marginalized are another tricky thing for this approach. In general we lose flexibility whenever the model is no longer iid over families, or more precisely when the sampling probability computed in the main likelihood routine no longer involves the assumption of iidness across the entire data set. If we would be able to take subgraphs corresponding to parts of the data efficiently, this approach might still enable speed-ups compared to computing partial likelihoods independently for each family.
