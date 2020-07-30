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
            Node(id(x), NodeProbs(x, m, T)) :
            Node(id(x), NodeProbs(x, m, T), y)
        for c in children(x) walk(c, y′) end
        push!(order, y′)
        return y′
    end
    n = walk(node, nothing)
    model = PhyloBDP(rates, Dict(id(n)=>n for n in order), order, m, cond)
    setmodel!(model)  # assume the model should be initialized
    return model
end

# These are two 'secondary constructors',
# i.e. they establish a model based on an already available model structure
# and a new set of parameters.
# The first makes a copy, the second modifies the existing model.
(m::PhyloBDP)(θ) = PhyloBDP(m.rates(θ), m.order[end], m.bound, cond=m.cond)
function update!(m::PhyloBDP, θ)
    m.rates = m.rates(θ)
    setmodel!(m)
end

setmodel!(model) = setmodel!(model.order, model.rates)
function setmodel!(order, rates)
    for n in order
        setϵ!(n, rates)
        setW!(n, rates)
    end
end

Base.getindex(m::PhyloBDP, i) = m.nodes[i]
Base.show(io::IO, m::PhyloBDP) = write(io, "PhyloBDP(\n~$(m.cond)\n$(m.rates))")
root(m::PhyloBDP) = m.order[end]
NewickTree.getroot(m::PhyloBDP) = root(m)

# NOTE that this does not involve the gain model!
function setϵ!(n::ModelNode{T}, rates::M) where {T,M<:Union{RatesModel,Params}}
    isleaf(n) && return  # XXX or should we set ϵ to 0.? [it should always be]
    θn = getθ(rates, n)
    if iswgd(n) || iswgt(n)
        c = first(children(n))
        ϵc = getϵ(c, 2)
        ϵn = iswgd(n) ? wgdϵ(θn.q, ϵc) : wgtϵ(θn.q, ϵc)
        setϵ!(c, 1, ϵn)
        setϵ!(n, 2, ϵn)
    else
        setϵ!(n, 2, one(T))
        for c in children(n)
            θc = getθ(rates, c)
            ϵc = extp(θc.λ, θc.μ, distance(c), getϵ(c, 2))
            setϵ!(c, 1, ϵc)
            setϵ!(n, 2, probify(getϵ(n, 2) * ϵc))
        end
    end
end

getϵ(n, i::Int) = n.data.ϵ[i]
setϵ!(n, i::Int, x) = n.data.ϵ[i] = x
wgdϵ(q, ϵ) = q*ϵ^2 + (one(q) - q)*ϵ
wgtϵ(q, ϵ) = q*ϵ^3 + 2q*(one(q) - q)*ϵ^2 + (one(q) - q)^2*ϵ

# non-linear models
function setW!(n::ModelNode{T}, rates) where T
    isroot(n) && return
    Q = getQ(rates.params, n)
    n.data.W .= exp(Q*distance(n))
end

# linear models (implementation detail technique)
function setW!(n::ModelNode{T}, rates::V) where {T,V<:LinearModel}
    isroot(n) && return
    ϵ = getϵ(n, 2)
    θ = getθ(rates, n)
    if iswgdafter(n)
        wstar_wgd!(n.data.W, distance(n), θ, ϵ)
    elseif iswgtafter(n)
        wstar_wgt!(n.data.W, distance(n), θ, ϵ)
    else
        wstar!(n.data.W, distance(n), θ, ϵ)
    end
    # NOTE: elseif isleaf -> take into account sampling probabilities?
    # tricky, because the bounds on the maximum state are no longer known in
    # the imperfect sampling setting, so we can only approximate this... (but
    # that is anyway the case also if we don't employ the CM algorithm)
end

Base.show(io::IO, x::ForwardDiff.Dual) = show(io, "dual$(x.value)")

function wstar!(w::Matrix{T}, t, θ, ϵ) where T  # compute w* (Csuros Miklos '09)
    @unpack λ, μ, κ = θ
    l = size(w)[1]-1
    ϕ = getϕ(t, λ, μ)  # p
    ψ = getψ(t, λ, μ)  # q
    _n = one(ψ) - ψ*ϵ
    ϕp = probify((ϕ*(one(ϵ) - ϵ) + (one(ψ) - ψ)*ϵ) / _n)
    ψp = probify(ψ*(one(ϵ) - ϵ) / _n)
    (κ/λ > zero(κ)) && (one(ψp) - ψp > zero(ψp)) ? # gain model
        w[1,:] = pdf.(NegativeBinomial(κ/λ, one(ψp) - ψp), 0:l) :
        w[1,1] = one(T)
    for m=1:l, n=1:m
        w[n+1, m+1] = ψp*w[n+1, m] + (one(ϕ) - ϕp)*(one(ψ) - ψp)*w[n, m]
    end
end

function wstar_wgd!(w, t, θ, ϵ)
    @unpack λ, μ, q = θ
    w[1,1] = one(q)
    w[2,2] = ((one(q) - q) + 2q*ϵ)*(one(ϵ) - ϵ)
    w[2,3] = q*(one(ϵ) - ϵ)^2
    l = size(w)[1]-1
    for i=1:l, j=2:l
        w[i+1, j+1] = w[2,2]*w[i, j] + w[2,3]*w[i, j-1]
    end
end

function wstar_wgt!(w, t, θ, ϵ)
    @unpack λ, μ, q = θ
    q1 = q
    q2 = 2q*(one(q) - q)
    q3 = (one(q) - q)^2
    w[1,1] = one(q)
    w[2,2] = q1*(one(q) - ϵ) + 2*q2*ϵ*(one(ϵ) - ϵ) + 3*q3*(ϵ^2)*(one(ϵ) - ϵ)
    w[2,3] = q2*(one(ϵ) - ϵ)^2 + 3q3*ϵ*(one(ϵ) - ϵ)^2
    w[2,4] = q3*(one(ϵ) - ϵ)^3
    l = size(w)[1]-1
    for i=1:l, j=3:l
        w[i+1, j+1] =  w[2,2]*w[i, j] + w[2,3]*w[i, j-1] + w[2,4]*w[i, j-2]
    end
end

const ΛMTOL = 1e-6
approx1(x) = x ≈ one(x) ? one(x) : x
approx0(x) = x ≈ zero(x) ? zero(x) : x

getϕ(t, λ, μ) = isapprox(λ, μ, atol=ΛMTOL) ?
    probify(λ*t/(one(λ) + λ*t)) :
    probify(μ*(exp(t*(λ-μ))-one(λ))/(λ*exp(t*(λ-μ))-μ))
getψ(t, λ, μ) = isapprox(λ, μ, atol=ΛMTOL) ?
    probify(λ*t/(one(λ) + λ*t)) :
    probify((λ/μ)*getϕ(t, λ, μ))
extp(λ, μ, t, ϵ=0.) = isapprox(λ, μ, atol=ΛMTOL) ?
    probify(one(ϵ) + (one(ϵ) - ϵ)/(μ * (ϵ - one(ϵ)) * t - one(ϵ))) :
    probify((μ+(λ-μ)/(one(ϵ)+exp((λ-μ)*t)*λ*(ϵ-one(ϵ))/(μ-λ*ϵ)))/λ)
getξ(i, j, k, t, λ, μ) = _bin(i, k)*_bin(i+j-k-1,i-1)*
    getϕ(t, λ, μ)^(i-k)*getψ(t, λ, μ)^(j-k)*(1-getϕ(t, λ, μ)-getψ(t, λ, μ))^k
tp(a, b, t, λ, μ) = (a == b == zero(a)) ? one(λ) :
    probify2(sum([getξ(a, b, k, t, λ, μ) for k=0:min(a,b)]))
logfact_stirling(n) = n*log(n) - n + log(2π*n)/2
_bin(n, k) = n > 60 ?
    exp(logfact_stirling(n) - logfact_stirling(k) - logfact_stirling(n - k)) :
    binomial(n, k)
probify2(p) = p > one(p) ? one(p) : p < zero(p) ? zero(p) : p

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

# using InteractiveUtils
# macro probify(p)
#     msg = string(p)
#     return :(if $p > one($p)
#             @warn "ℙ > 1 ($($msg) = $($p))"
#             one($p)
#         elseif $p < zero($p)
#             @warn "ℙ < 0 ($($msg) = $($p))"
#             zero($p)
#         else
#             $p
#         end)
#     end
# end

"""
    ∫rootgeometric(ℓ, η, ϵ)

Integrate the loglikelihood at the root according to a shifted
geometric prior with mean 1/η and log extinction probablity ϵ.
Assumes at least one ancestral gene.
"""
@inline function ∫rootgeometric(ℓ, η, lϵ)
    p = -Inf
    for i in 2:length(ℓ)
        f = (i-1)*log1mexp(lϵ) + log(η) + (i-2)*log(1. - η)
        f -= i*log1mexp(log(one(η) - η)+lϵ)
        p = logaddexp(p, ℓ[i] + f)
    end
    return p
end

# We could work with types as well and use dispatch...
conditionfactor(model) =
    if model.cond == :root
        nonextinctfromrootcondition(model)
    elseif model.cond == :nowhere
        extinctnowherecondition(model)
    else
        0.
end

# This is the non-extinction in both clades stemming from the root condition
# XXX: assumes the geometric prior!
function nonextinctfromrootcondition(model)
    @unpack η = getθ(model.rates, model[1])
    lη = log(η)
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

# XXX: see pgf technique for the stuff below! (implemented in Whale)
# NOTE: experimental, will not work OOTB with WGDs. Also, will not work with gain model. All doable though. First I was trying to compute the probability of extinction somewhere, but the probability of extinction nowhere turned out to be more easily calculated in a preorder, much like one would simulate from a CTMC time with finite state space. This is of course approximate!
# function extinctnowherecondition(m::PhyloBDP{T}, bound=m.bound*2) where T
#     𝑃 = zeros(T, bound, length(m.order))
#     p = one(T)
#     function walk(n)
#         _pvec!(𝑃, m, n)
#         for c in children(n) walk(c) end
#         if isleaf(n)
#             p *= sum(𝑃[2:end, id(n)])
#         end
#         return
#     end
#     walk(root(m))
#     return log(probify(p))
# end
#
# function _pvec!(𝑃, model, n)
#     if isroot(n)
#         @unpack η = getθ(model.rates, n)
#         𝑃[:,id(n)] = [0. ; pdf.(Geometric(η), 0:size(𝑃)[1]-2)]
#     else
#         @unpack λ, μ = getθ(model.rates, n)
#         t = distance(n)
#         bound = size(𝑃)[1]
#         matrix = [tp(i, j, t, λ, μ) for i=0:bound-1, j=0:bound-1]
#         p = matrix' * 𝑃[:,id(parent(n))]
#         𝑃[:,id(n)] .= p /sum(p)
#     end
# end
