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

"""
Simple recursive simulator for DL(WGD) model with a homogeneous
rate for each branch. Currently assumes the Geometric prior on
the number of lineages at the root. Note that gain events are
not simulated, so if a DLG model is provided, the gain rate will
be ignored.
"""
dlsim(Ψ, θ) = dlsim(Ψ, θ, rand(Geometric(θ.params.η))+1)
function dlsim(Ψ, θ, a::Int)
    sproot = getroot(Ψ)
    ns = [dlsim!(Node(0, Rec(id(sproot), 0., "")), 0., sproot, θ) for i=1:a]
    root = Node(0, Rec(id(sproot), 0., a>1 ? "dup" : "sp"))
    for n in ns push!(root, n) end
    label!(root)
    return root
end

function dlsim!(n, t, e, model) where I
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

dlsimbunch(Ψ, N; kwargs...) = dlsimbunch(getroot(Ψ), Ψ.rates, N; kwargs...)
function dlsimbunch(Ψ, model, N; condition=:none, minn=3)
    if condition == :root
        condition = [name.(getleaves(Ψ[1])), name.(getleaves(Ψ[2]))]
    elseif condition == :all
        condition = [[name(x)] for x in getleaves(Ψ)]
    elseif condition == :none
        condition = [name.(getleaves(Ψ))]
    end
    n = 0; m = 0
    leaves = name.(getleaves(Ψ))
    tree = dlsim(Ψ, model)
    trees = typeof(tree)[]
    profiles = DataFrame()
    while length(trees) < N
        t = dlsim(Ψ, model)
        pruneloss!(t)
        p = profile(t, leaves)
        l = length(getleaves(t))
        if !all([any(x->p[x] > 0, xs) for xs in condition]) || l < minn
            n += 1
            all(x->x==0, values(p)) ? m += 1 : nothing
            continue
        end
        profiles = vcat(profiles, DataFrame(p))
        push!(trees, t)
    end
    @debug "$n trees violated condition, of which $m were completely extinct"
    return (trees=trees, profiles=profiles, violated=n/N)
end
