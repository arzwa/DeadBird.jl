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
iswgd(n) = name(n) == "wgd"
iswgt(n) = name(n) == "wgt"
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

# these are two 'secondary constructors', i.e. they establish a model based on an already available modal structure and a new set of parameters. The first makes a copy, the second modifies the existing model.
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

function setW!(n::ModelNode{T}, rates::M) where {T,M<:Union{RatesModel,Params}}
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

function wstar!(w, t, θ, ϵ)  # compute w* (Csuros Miklos 2009)
    @unpack λ, μ, κ = θ
    l = size(w)[1]-1
    ϕ = getϕ(t, λ, μ)  # p
    ψ = getψ(t, λ, μ)  # q
    _n = one(ψ) - ψ*ϵ
    ϕp = probify((ϕ*(one(ϵ) - ϵ) + (one(ψ) - ψ)*ϵ) / _n)
    ψp = probify(ψ*(one(ϵ) - ϵ) / _n)
    κ/λ > zero(κ) && one(ψp) - ψp > zero(ψp) ? # gain model
        w[1,:] = pdf.(NegativeBinomial(κ/λ, one(ψp) - ψp), 0:l) :
        w[1,1] = one(ψ)
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
    probify(sum([getξ(a, b, k, t, λ, μ) for k=0:min(a,b)]))
logfact_stirling(n) = n*log(n) - n + log(2π*n)/2
_bin(n, k) = n > 60 ?
    exp(logfact_stirling(n) - logfact_stirling(k) - logfact_stirling(n - k)) :
    binomial(n, k)

# maybe make this a macro, so that we can show the function call?
function probify(p)
    return if p > one(p)
        @warn "probability $p > 1, set to 1"
        one(p)
    elseif p < zero(p)
        @warn "probability $p < 0, set to 0"
        zero(p)
    else
        p
    end
end

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

# We could work with tyes as well and use dispatch...
conditionfactor(model) =
    if model.cond == :root
        nonextinctfromrootcondition(model)
    elseif model.cond == :nowhere
        extinctnowherecondition(model)
    else
        0.
end

# This is the non-extinction in both clades stemming from the root condition
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

# NOTE: experimental, will not work OOTB with WGDs. Also, will not work with gain model. Not sure whether it actually works... The values tend to make sense, but the algorithms seems to be numerically unstable...
function extinctnowherecondition(model::PhyloBDP{T},
        bound=2*model.bound) where T
    # we compute the probability of extinction *somewhere* recursively
    𝑃 = zeros(T, bound, length(model.order))
    # 𝑃 stores the state probabilities (for nonextinct states!)
    function walk(n)
        isleaf(n) && return 0.
        _pvec!(𝑃, model, n) # TO DO
        qs = zeros(T, length(children(n)))
        ps = zeros(T, length(children(n)))
        for (i,c) in enumerate(children(n))
            @unpack λ, μ = getθ(model.rates, c)
            qc = [extp(λ, μ, distance(c))^i for i=1:bound]
            qs[i] = qc'*𝑃[:,id(n)]
            ps[i] = walk(c) # recurse here
        end
        # Now be careful to do inclusion/exclusion correctly when not binary
        ps′ = [(1. - qs[i])*ps[i] for i=1:length(qs)]
        p = inclusion_exclusion(qs) + inclusion_exclusion(ps′)
        # return a probability
        return p
    end
    p = probify(walk(root(model)))
    log(one(p) - p)
end

function _pvec!(𝑃, model, n)
    if isroot(n)
        @unpack η = getθ(model.rates, n)
        𝑃[:,id(n)] = pdf.(Geometric(η), 0:size(𝑃)[1]-1)
    else
        @unpack λ, μ = getθ(model.rates, n)
        t = distance(n)
        bound = size(𝑃)[1]
        matrix = [tp(i, j, t, λ, μ) for i=1:bound, j=1:bound]
        # matrix = hcat(eachrow(matrix) ./ eachrow(sum(matrix, dims=2))...)'
        # XXX should we renormalize the matrix or not, no, the vecctor should
        p = matrix' * 𝑃[:,id(parent(n))]
        𝑃[:,id(n)] .= p /sum(p)
    end
end

function inclusion_exclusion(p)
    s = sum(p)
    for i=2:length(p), combo in eachcol(combinations(length(p), i))
        s -= prod([p[j] for j in combo])
    end
    s
end

function combinations(n, r)
    function walk(partial)
        length(partial) == r && return partial
        j = partial[end]+1
        partials = [walk(vcat(partial, i)) for i=j:n]
        return hcat([p for p in partials if length(p) > 0]...)
    end
    ps = [walk([i]) for i=1:n-r+1]
    return hcat([p for p in ps if length(p)>0]...)
end
