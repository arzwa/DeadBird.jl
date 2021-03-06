# Model nodes for the PGM
# hate the name, but'it's barely used (the name that is)
struct NodeProbs{T}
    name::String  # leaf name
    k::Int        # multiplication level
    t::Float64    # usually distances have a fixed type
    ϵ::Vector{T}  # extinction probabilities
    W::Matrix{T}  # these are upper-triangular, but on log-scale...
end

function NodeProbs(n, m::Int, ::Type{T}) where T
    NodeProbs(name(n), getk(n), distance(n), m, T)
end

function NodeProbs(name, k, t, m::Int, ::Type{T}) where T
    ϵ = fill(T(-Inf), 2)
    W = fill(T(-Inf), m, m)
    NodeProbs(name, k, t, ϵ, W)
end

Base.show(io::IO, n::NodeProbs) = write(io, "$(name(n)), k=$(n.k)")

NewickTree.name(n::NodeProbs) = n.name
NewickTree.distance(n::NodeProbs) = n.t

# this is used more often
const ModelNode{T,I} = Node{I,NodeProbs{T}}
getbound(m::ModelNode) = size(m.data.W, 1)

# extinction probabilities
getϵ(n, i::Int) = n.data.ϵ[i]
setϵ!(n, i::Int, x) = n.data.ϵ[i] = x

# multiplication levels, the first applies to the nodes in the input tree
getk(n) = 1  # could be changed to check for `name(n) == "wgd"` or so
getk(n::ModelNode) = n.data.k
iswgm(n) = getk(n) > 1
iswgd(n) = getk(n) == 2
iswgt(n) = getk(n) == 3
iswgmafter(n) = iswgm(parent(n))
iswgdafter(n) = iswgd(parent(n))
iswgtafter(n) = iswgt(parent(n))
isawgm(n) = iswgm(n) || iswgm(parent(n))

# gives the index for this wgd, for now we assume its id(n), i.e. we so that
# the retention rate should be obtained from element id(n) in the q-vector of
# the rates model. This entails the q-vector is not as compact as it should,
# but this is very convenient.
wgmid(n) = id(n) 

"""
    PhyloBDP(ratesmodel, tree, bound)

The phylogenetic birth-death process model as defined by Csuros & Miklos
(2009). The bound is exactly defined by the data under consideration.

!!! note: implemented as a `<: DiscreteMultivariateDistribution` (for
    convenience with Turing.jl), however does not support a lot of the
    Distributions.jl interface.

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
struct PhyloBDP{T,M,P,I} <: DiscreteMultivariateDistribution
    rates::M
    rootp::P
    nodes::Dict{I,ModelNode{T,I}}  # redundant, but convenient
    order::Vector{ModelNode{T,I}}
    bound::Int
    cond ::Symbol
end

const LPhyloBDP{T} = PhyloBDP{T,M} where {T,M<:LinearModel}

# We assume the types of the RatesModel
function PhyloBDP(rates :: RatesModel{T}, 
                  rootp :: RootPrior, 
                  node  :: Node{I}, 
                  bound :: Int; 
                  cond  :: Symbol = :root) where {T,I}
    order = ModelNode{T,I}[]
    nodes = Dict{I,eltype(order)}()
    # x is the node to be copied, y is the parent of copy to created
    function walk(x, y)
        y′ = Node(id(x), NodeProbs(x, bound+1, T), y)
        for c in children(x) walk(c, y′) end
        push!(order, y′)
        nodes[id(y′)] = y′
    end
    walk(node, nothing)
    model = PhyloBDP(rates, rootp, nodes, order, bound+1, cond)
    setmodel!(model)  # assume the model should be initialized
    return model
end

# handy constructor based on pre-existing model
function (m::PhyloBDP)(; rates=m.rates, rootp=m.rootp, bound=m.bound-1)
    PhyloBDP(rates, rootp, m.order[end], bound, cond=m.cond)
end

function Base.show(io::IO, m::PhyloBDP) 
    write(io, "PhyloBDP(\n"*"condition: $(m.cond)\n")
    write(io, "bound: $(m.bound) \n$(m.rates))")
end

Base.getindex(m::PhyloBDP, i) = m.nodes[i]
Base.length(m::PhyloBDP) = length(m.order)

root(m::PhyloBDP) = m.order[end]
NewickTree.getroot(m::PhyloBDP) = root(m)
rootprior(m::PhyloBDP) = m.rootp

struct ModelArray{M} <: DiscreteMultivariateDistribution
    models::Vector{M}
end

Base.getindex(m::ModelArray, i) = m.models[i]
Base.length(m::ModelArray) = length(m.models)

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

