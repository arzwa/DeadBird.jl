# Arthur Zwaenepoel (2020)
# Consider the following design for arbitrary multiplication nodes:
# instead of relying on a string/symbol/type to identify WGDs, we just
# associate with each node a multiplication level `k`, if `k == 1` we have a
# speciation node (bi/multi-furcating)  or `wgdafter` node (non- bifurcating
# case). For `k > 1` we have a multiplication node...

# hate the name, but'it's barely used (the name that is)
struct NodeProbs{T}
    name::String  # leaf name
    k::Int        # multiplication level
    t::Float64    # usually distances have a fixed type
    ϵ::Vector{T}
    W::Matrix{T}
end

function NodeProbs(n, m::Int, k::Int, ::Type{T}) where T
    ϵ = fill(T(-Inf), 2)
    W = fill(T(-Inf), m, m)
    NodeProbs(name(n), k, distance(n), ϵ, W)
end

Base.show(io::IO, n::NodeProbs) = write(io, "$(name(n)), k=$(n.k)")
NewickTree.name(n::NodeProbs) = n.name
NewickTree.distance(n::NodeProbs) = n.t

# this is used more often
const ModelNode{T,I} = Node{I,NodeProbs{T}}

# extinction probabilities
getϵ(n, i::Int) = n.data.ϵ[i]
setϵ!(n, i::Int, x) = n.data.ϵ[i] = x

# multiplication levels
getk(n) = n.data.k
iswgd(n) = getk(n) == 2
iswgt(n) = getk(n) == 3
iswgdafter(n) = iswgd(parent(n))
iswgtafter(n) = iswgt(parent(n))

"""
    PhyloBDP(ratesmodel, tree, bound)

The phylogenetic birth-death process model as defined by Csuros &
Miklos (2009). The bound is exactly defined by the data under
consideration.

!!! note: implemented as a `<: DiscreteMultivariateDistribution`
    (for convenience with Turing.jl), however does not support
    a lot of the Distributions.jl interface.

# Example
```julia-repl
julia> x = DeadBird.example_data();

julia> rates = RatesModel(ConstantDLG(λ=0.1, μ=0.1));

julia> dag, bound = CountDAG(x.df, x.tr);

julia> rates = RatesModel(ConstantDLG(λ=0.1, μ=0.1));

julia> PhyloBDP(rates, x.tr, bound)
PhyloBDP(
~root
RatesModel with () fixed
ConstantDLG{Float64}
  λ: Float64 0.1
  μ: Float64 0.1
  κ: Float64 0.0
  η: Float64 0.66
)
```
"""
mutable struct PhyloBDP{T,M,I} <: DiscreteMultivariateDistribution
    rates::M
    nodes::Dict{I,ModelNode{T,I}}  # stored in postorder, redundant
    order::Vector{ModelNode{T,I}}
    bound::Int
    cond ::Symbol
end

const LPhyloBDP{T} = PhyloBDP{T,V} where {T,V<:LinearModel}

function PhyloBDP(rates::RatesModel{T}, node::Node{I}, m::Int; 
                  cond::Symbol=:root) where {T,I}
    order = ModelNode{T,I}[]
    function walk(x, y)
        y′ = isroot(x) ?
            Node(id(x), NodeProbs(x, m+1, 1, T)) :
            Node(id(x), NodeProbs(x, m+1, 1, T), y)
        for c in children(x) walk(c, y′) end
        push!(order, y′)
        return y′
    end
    n = walk(node, nothing)
    model = PhyloBDP(rates, Dict(id(n)=>n for n in order), order, m+1, cond)
    setmodel!(model)  # assume the model should be initialized
    return model
end

function Base.show(io::IO, m::PhyloBDP) 
    write(io, "PhyloBDP(\n"*"condition: $(m.cond)\n")
    write(io, "bound: $(m.bound) \n$(m.rates))")
end

Base.getindex(m::PhyloBDP, i) = m.nodes[i]
root(m::PhyloBDP) = m.order[end]
NewickTree.getroot(m::PhyloBDP) = root(m)

struct ModelArray{M} <: DiscreteMultivariateDistribution
    models::Vector{M}
end

Base.getindex(m::ModelArray, i) = m.models[i]
Base.length(m::ModelArray) = length(m.models)

# These are two 'secondary constructors',
# i.e. they establish a model based on an already available model structure
# and a new set of parameters.
# The first two make a copy (the second only adapts the bound), while the last
# modifies the existing model.
(m::PhyloBDP)(θ, b=m.bound-1) = PhyloBDP(m.rates(θ), m.order[end], b, cond=m.cond)
(m::PhyloBDP)(b::Int) = PhyloBDP(m.rates, m.order[end], b, cond=m.cond)
function update!(m::PhyloBDP, θ)
    m.rates = m.rates(θ)
    setmodel!(m)
end

function setmodel!(model::LPhyloBDP)
    @unpack order, rates = model
    for n in order
        setϵ!(n, rates)
        setW!(n, rates)
    end
end

# loglikelihood -> logpdf
Distributions.loglikelihood(m::PhyloBDP, x) = logpdf(m, x)
Distributions.loglikelihood(M::ModelArray, x) = logpdf(M, x)
function Distributions.loglikelihood(
        m::MixtureModel{Multivariate,Discrete,P}, x) where P<:PhyloBDP 
    return logpdf(m, x)
end

