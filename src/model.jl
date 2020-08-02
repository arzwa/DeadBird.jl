# Split this file in model.jl and linear.jl
# hate the name, but'it's barely used (the name that is)
struct NodeProbs{T}
    name::String  # leaf name/wgd/wgt ...
    t::Float64    # usually distances have a fixed type
    ϵ::Vector{T}
    W::Matrix{T}
end

const ModelNode{T,I} = Node{I,NodeProbs{T}}
NodeProbs(n, m::Int) = NodeProbs(name(n), distance(n), zeros(2), zeros(m,m))
NodeProbs(n, m::Int, T::Type) =
    NodeProbs(name(n), distance(n), zeros(T, 2), zeros(T, m,m))
Base.show(io::IO, n::NodeProbs{T}) where T = write(io, n.name)
NewickTree.name(n::NodeProbs) = n.name
NewickTree.distance(n::NodeProbs) = n.t
iswgd(n) = startswith(name(n), "wgd")
iswgt(n) = startswith(name(n), "wgt")
wgdid(n) = iswgd(n) ? parse(Int64, split(name(n))[2]) : NaN
iswgdafter(n) = name(n) == "wgdafter"
iswgtafter(n) = name(n) == "wgtafter"

"""
    PhyloBDP(ratesmodel, tree, bound)

The phylogenetic birth-death process model as defined by Csuros &
Miklos (2009). The bound is exactly defined by the data under
consideration.

!!! note: implemented as a `<: DiscreteMultivariateDistribution`
    (for convenience with Turing.jl), however does not support
    a lot of the Distributions.jl interface.
"""
mutable struct PhyloBDP{T,M,I} <: DiscreteMultivariateDistribution
    rates::M
    nodes::Dict{I,ModelNode{T,I}}  # stored in postorder, redundant
    order::Vector{ModelNode{T,I}}
    bound::Int
    cond ::Symbol
end

function PhyloBDP(rates::RatesModel{T}, node::Node{I}, m::Int;
        cond::Symbol=:root) where {T,I}
    order = ModelNode{T,I}[]
    function walk(x, y)
        y′ = isroot(x) ?
            Node(id(x), NodeProbs(x, m+1, T)) :
            Node(id(x), NodeProbs(x, m+1, T), y)
        for c in children(x) walk(c, y′) end
        push!(order, y′)
        return y′
    end
    n = walk(node, nothing)
    model = PhyloBDP(rates, Dict(id(n)=>n for n in order), order, m+1, cond)
    setmodel!(model)  # assume the model should be initialized
    return model
end

# These are two 'secondary constructors',
# i.e. they establish a model based on an already available model structure
# and a new set of parameters.
# The first makes a copy, the second modifies the existing model.
(m::PhyloBDP)(θ) = PhyloBDP(m.rates(θ), m.order[end], m.bound-1, cond=m.cond)
function update!(m::PhyloBDP, θ)
    m.rates = m.rates(θ)
    setmodel!(m)
end

Base.getindex(m::PhyloBDP, i) = m.nodes[i]
Base.show(io::IO, m::PhyloBDP) = write(io, "PhyloBDP(\n~$(m.cond)\n$(m.rates))")
root(m::PhyloBDP) = m.order[end]
NewickTree.getroot(m::PhyloBDP) = root(m)

# maybe make this a macro, so that we can show the function call?
const PTOL = 1e-9  # tolerance for probabilities
function probify(p)
    return if p > one(p)
        !(isapprox(p, one(p), atol=PTOL)) && @warn "probability $p > 1, set to 1"
        one(p)
    elseif p < zero(p)
        !(isapprox(p, zero(p), atol=PTOL)) && @warn "probability $p < 0, set to 0"
        zero(p)
    else
        p
    end
end
