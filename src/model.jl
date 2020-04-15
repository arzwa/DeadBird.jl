# The RatesModel interface implements two functions: getθ(ratesmodel, node) to obtain the parameters associated with a node in the model and the function (::ratesmodel)(params) to obtain a new properly typed model object from a pre-existing one with new parameters

# As in the Whale implementation, we would also want a transformation to be defined. That would define all necessary interfacing for doing optimization and NUTS?

# IDEA: define a WGD model as a wrapper around a RatesModel.

# XXX: think about how we could efficiently 'fix' parameters.

abstract type RatesModel{T} end

"""
    ConstantDLG{T}

Simple constant rates duplication-loss and gain model. All nodes of
the tree are associated with the same parameters (duplication rate λ,
loss rate μ, gain rate κ). This assumes a shifted geometric distribution
on the family size at the root with mean 1/η.
"""
@with_kw struct ConstantDLG{T} <: Params{T}
    λ::T
    μ::T
    κ::T = 0.
    η::T = 0.66
end

getθ(m::ConstantDLG, node) = m
trans(::ConstantDLG) = as((λ=asℝ₊, μ=asℝ₊, κ=asℝ₊, η=as𝕀))
(::ConstantDLG)(θ) = ConstantDLG(promote(θ...)...)

function Base.rand(m::M) where M<:RatesModel
    t = trans(m)
    M(;t(randn(dimension(t)))...)
end

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
"""
struct PhyloBDP{T,M,I}
    rates::M
    nodes::Dict{I,ModelNode{T,I}}  # stored in postorder, redundant
    order::Vector{ModelNode{T,I}}
end

function PhyloBDP(rates::RatesModel{T}, node::Node, m::Int) where T<:Real
    function walk(x, y)
        y′ = isroot(x) ?
            Node(id(x), NodeProbs(x, m, T)) :
            Node(id(x), NodeProbs(x, m, T), y)
        for c in children(x) walk(c, y′) end
        return y′
    end
    n = walk(node, nothing)
    order = postwalk(n)
    model = PhyloBDP(rates, Dict(id(n)=>n for n in order), order)
    setmodel!(model)
    return model
end

function setmodel!(model)
    @unpack order, rates = model
    for n in order
        setϵ!(n, rates)
        setW!(n, rates)
    end
end

Base.getindex(m::PhyloBDP, i) = m.nodes[i]
Base.show(io::IO, m::PhyloBDP) = write(io, "PhyloBDP(\n$(m.rates))")

# NOTE that this does not involve the gain model!
function setϵ!(n::ModelNode{T}, rates::M) where {T,M<:RatesModel}
    isleaf(n) && return  # XXX or should we set ϵ to 0.? [it should always be]
    θ = getθ(rates, n)
    if iswgd(n) || iswgt(n)
        c = first(children(n))
        ϵc = getϵ(c, 2)
        ϵn = iswgd(n) ? wgdϵ(θ.q, ϵc) : wgtϵ(θ.q, ϵc)
        setϵ!(c, 1, ϵn)
        setϵ!(n, 2, ϵn)
    else
        setϵ!(n, 2, one(T))
        for c in children(n)
            ϵc = approx1(extp(θ.λ, θ.μ, distance(c), getϵ(c, 2)))
            setϵ!(c, 1, ϵc)
            setϵ!(n, 2, getϵ(n, 2) * ϵc)
        end
        setϵ!(n, 2, approx1(getϵ(n, 2)))
    end
end

getϵ(n, i::Int) = n.data.ϵ[i]
setϵ!(n, i::Int, x) = n.data.ϵ[i] = x
wgdϵ(q, ϵ) = q*ϵ^2 + (one(q) - q)*ϵ
wgtϵ(q, ϵ) = q*ϵ^3 + 2q*(one(q) - q)*ϵ^2 + (one(q) - q)^2*ϵ

function setW!(n::ModelNode{T}, rates::M) where {T,M<:RatesModel}
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

# adapt for gain... + where can we use matrix expressions?
function wstar!(w, t, θ, ϵ)  # compute w* (Csuros Miklos 2009)
    @unpack λ, μ, κ = θ
    l = size(w)[1]-1
    ϕ = getϕ(t, λ, μ)  # p
    ψ = getψ(t, λ, μ)  # q
    _n = one(ψ) - ψ*ϵ
    ϕp = approx1((ϕ*(one(ϵ) - ϵ) + (one(ψ) - ψ)*ϵ) / _n)
    ψp = approx1(ψ*(one(ϵ) - ϵ) / _n)
    κ > zero(κ) ? # gain model
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

approx1(x) = x ≈ one(x) ? one(x) : x
approx0(x) = x ≈ zero(x) ? zero(x) : x

getϕ(t, λ, μ) = λ ≈ μ ?
    λ*t/(one(λ) + λ*t) :
    μ*(exp(t*(λ-μ))-one(λ))/(λ*exp(t*(λ-μ))-μ)
getψ(t, λ, μ) = λ ≈ μ ?
    λ*t/(one(λ) + λ*t) :
    (λ/μ)*getϕ(t, λ, μ)
extp(λ, μ, t, ϵ) = λ ≈ μ ?
    one(ϵ) + (one(ϵ) - ϵ)/(μ * (ϵ - one(ϵ)) * t - one(ϵ)) :
    approx1((μ+(λ-μ)/(one(ϵ)+exp((λ-μ)*t)*λ*(ϵ-one(ϵ))/(μ-λ*ϵ)))/λ)

# This is the 'classical' implementation, operating on a single (extended) profile NOTE: possible optimizations: (1) matrix operations instead of some loops (not likely to improve speed?)
@inline function cm!(
        L::Matrix{T},
        x::Vector{Int64},
        n::ModelNode{T}) where T<:Real
    # @unpack W, ϵ = n.data
    xmax = maximum(x)
    if isleaf(n)
        L[x[id(n)]+1, id(n)] = 0.
    else
        kids = children(n)
        cmax = [x[id(c)] for c in kids]
        ccum = cumsum([0 ; cmax])
        ϵcum = cumprod([1.; [getϵ(c, 1) for c in kids]])
        # XXX possible numerical issues with ϵcum?
        B = fill(-Inf, (xmax+1, ccum[end]+1, length(cmax)))
        A = fill(-Inf, (ccum[end]+1, length(cmax)))
        for i = 1:length(cmax)
            c  = kids[i]
            mi = cmax[i]
            Wc = c.data.W[1:xmax+1, 1:xmax+1]
            @inbounds B[:, 1, i] = log.(Wc * exp.(L[1:xmax+1, id(c)]))
            ϵ₁ = log(getϵ(c, 1))
            for t=1:ccum[i], s=0:mi  # this is 0...M[i-1] & 0...mi
                @inbounds B[s+1,t+1,i] = s == mi ?
                    B[s+1,t,i] + ϵ₁ : logaddexp(B[s+2,t,i], ϵ₁+B[s+1,t,i])
            end
            if i == 1
                l1me = log(one(ϵ₁) - ϵcum[2])
                for n=0:ccum[i+1]  # this is 0 ... M[i]
                    @inbounds A[n+1,i] = B[n+1,1,i] - n*l1me
                end
            else
                # XXX is this loop as efficient as it could? I guess not...
                p = probify(ϵcum[i])
                for n=0:ccum[i+1], t=0:ccum[i]
                    s = n-t
                    (s < 0 || s > mi) && continue
                    @inbounds lp = binomlogpdf(n, p, s) +
                        A[t+1,i-1] + B[s+1,t+1,i]
                    @inbounds A[n+1,i] = logaddexp(A[n+1,i], lp)
                end
                l1me = log(one(ϵ₁) - ϵcum[i+1])
                for n=0:ccum[i+1]  # this is 0 ... M[i]
                    @inbounds A[n+1,i] -= n*l1me
                end
            end
        end
        # @show A[:,end]
        # if not filling in a matrix, A[:,end] should be the output vector I
        # guess. The length of this vector would simultaneously specify the
        # maximum bound for the node
        for i=0:x[id(n)]
            @inbounds L[i+1, id(n)] = A[i+1,end]
        end
    end
end

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

# NOTE: not to be used when analyzing multiple profiles! (setϵ and setW should
# be invoked only once per set of parameters.)
function compute_conditionals!(L, x, m)
    for n in m.order
        setϵ!(n, m.rates)
        setW!(n, m.rates)
        cm!(L, x, n)
    end
end
