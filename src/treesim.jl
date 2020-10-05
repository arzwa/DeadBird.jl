
# copied from Whale...
randexp(λ) = -log(rand())/λ

mutable struct Rec{I,T}
    e::I
    d::T
    l::String
end
NewickTree.distance(r::Rec) = round(r.d, digits=6)
NewickTree.support(r::Rec) = NaN
NewickTree.name(n::Node{I,<:Rec}) where I =
    isleaf(n) ? "$(n.data.l)_$(n.data.l)$(id(n))" : ""

getlabel(n::Node{I,<:Rec}) where I = n.data.l
label!(r) = for (i, x) in enumerate(prewalk(r)) x.id = i end

#Base.rand(model::PhyloBDP, leaves=name.(getleaves(getroot(model)))) =
#    randprofile(model, leaves)

function randprofile(model::LPhyloBDP, leaves)
    t = randtree(model)
    pruneloss!(t)
    profile(t, leaves)
end

"""
    randtree(model::PhyloBDP)
    randtree(tree, model::RatesModel)

Simple recursive tree simulator for DL(WGD) model with a homogeneous
rate for each branch. Currently assumes the shifted geometric prior on
the number of lineages at the root. Note that gain events are
not simulated, so if a DLG model is provided, the gain rate will
be ignored.
"""
randtree(model) = randtree(model, model.rates)
randtree(Ψ, θ)  = randtree(Ψ, θ, rand(Geometric(θ.params.η))+1)

function randtree(Ψ, θ, a::Int)
    sproot = getroot(Ψ)
    ns = [dlsim!(Node(0, Rec(id(sproot), 0., "")), 0., sproot, θ) for i=1:a]
    root = Node(0, Rec(id(sproot), 0., a>1 ? "dup" : "sp"))
    for n in ns push!(root, n) end
    label!(root)
    return root
end

# Linear models
function dlsim!(n, t, e, model::LinearModel)
    @unpack λ, μ = getθ(model, e)
    w = randexp(λ+μ)
    t -= w
    if t > zero(t)
        n.data.d += w
        if rand() < λ/(λ+μ)  # dup
            n.data.l = "dup"
            dlsim!(Node(0, Rec(id(e), 0., ""), n), t, e, model)
            dlsim!(Node(0, Rec(id(e), 0., ""), n), t, e, model)
        else  # loss
            n.data.l = "loss"
            return
        end
    else
        n.data.d += t + w
        n.data.l = isleaf(e) ? name(e) : "sp"
        # if next is wgd -> wgd model
        if iswgd(e)
            @unpack q = getθ(model, e)
            n.data.l = "wgd"
            d = distance(e[1])
            if rand() < q  # retention
                dlsim!(Node(0, Rec(id(e[1]), 0., ""), n), d, e[1], model)
                dlsim!(Node(0, Rec(id(e[1]), 0., ""), n), d, e[1], model)
            else  # non-retention
                dlsim!(Node(0, Rec(id(e[1]), 0., ""), n), d, e[1], model)
            end
        else
            for c in children(e)
                dlsim!(Node(0, Rec(id(c), 0., ""), n), distance(c), c, model)
            end
        end
    end
    return n
end

pruneloss(tree) = pruneloss!(deepcopy(tree))
function pruneloss!(n)
    for node in postwalk(n)
        isroot(node) && return node
        l = getlabel(node)
        if isleaf(node) && l ∈ ["loss", "dup", "sp", "wgd"]
            delete!(parent(node), node)
            node.parent = node
        elseif NewickTree.degree(node) == 1
            p = parent(node)
            c = node[1]
            c.data.d += distance(node)
            delete!(p, node)
            push!(p, c)
            c.parent = p
        end
    end
    return n
end

profile(t, Ψ) = profile(t, name.(getleaves(Ψ)))
function profile(t, sleaves::Vector)
    m = countmap(getlabel.(getleaves(t)))
    Dict(k=>haskey(m, k) ? m[k] : 0 for k in sleaves)
end

function dlsimbunch(model, N; condition=:none, minn=3)
    o = getroot(model)
    if condition == :root
        condition = [name.(getleaves(o[1])), name.(getleaves(o[2]))]
    elseif condition == :all
        condition = [[name(x)] for x in getleaves(o)]
    elseif condition == :none
        condition = [name.(getleaves(o))]
    end
    accept(p, l) = all([any(x->p[x] > 0, xs) for xs in condition]) && l >= minn
    n = m = 0
    leaves = name.(getleaves(o))
    out = map(1:N) do i
        p = undef
        while true
            p = randprofile(model, leaves)
            l = sum(values(p))
            accept(p, l) ? break : n += 1
        end
        DataFrame(p)
    end
    (violated=n/N, profiles=vcat(out...))
end

# Vector of named tuples -> named tuple of vectors
# # (based on the field names in the first element)
# function group(X::Vector{<:NamedTuple})
#     map(keys(X[1])) do f
#         (;f=>mapreduce(x->getfield(x,f), vcat, X),)
#     end
# end

# Simulate profiles directly using truncated model
function randprofile(model, leaves)
    profile = Dict{String,Int}(s=>0 for s in leaves)
    function walk(n, x)
        if isleaf(n)
            profile[name(n)] = x
            return
        end
        for c in children(n)
            p = c.data.W[x+1,:]
            x′ = sample(Weights(p)) - 1
            walk(c, x′)
        end
    end
    xₒ = Int(rand(truncated(Geometric(model.rates.params.η), 1, model.bound-1)))
    walk(getroot(model), xₒ)
    return profile
end
