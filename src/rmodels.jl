# module RatesModels
# using Parameters, TransformVariables
# think think think, also make it a submodule
# A gene family evolution mode always consists of:
# 1. a model of DL rates across the tree (contant, local, branch-wise)
# 2. a model of across faily variation (Gamma mixture, DP mixture, ...)
# 3. a prior for the root state

abstract type Params{T} end

struct RatesModel{T,M<:Params{T},V}
    params::M
    fixed ::Tuple
    trans ::V
end

Base.eltype(m::RatesModel{T}) where T = T
RatesModel(θ; fixed=()) = RatesModel(θ, fixed, gettrans(θ, fixed))
Base.show(io::IO, m::RatesModel) = write(io,
    "RatesModel with $(m.fixed) fixed\n$(m.params)")
getθ(m::RatesModel, node) = getθ(m.params, node)

# HACK: a little bit of metaprogramming to allow fixed parameters, necessary?
function gettrans(p::P, fixed) where P<:Params
    inner = join(["$k=$v," for (k,v) in pairs(trans(p)) if k ∉ fixed])
    expr  = Meta.parse("as(($inner))")
    eval(expr)
end

(m::RatesModel)(x::Vector) = m(m.trans(x))
function (m::RatesModel)(θ)
    θ′ = merge(θ, [k=>getparam(m.params, k) for k in m.fixed])
    RatesModel(m.params(θ′), m.fixed, m.trans)
end

getparam(m::Params, v) =  getfield(m, v)

Base.rand(m::M) where M<:RatesModel = m(m.trans(randn(dimension(m.trans))))

promote_nt(nt) = (;zip(keys(nt), promote(nt...))...)

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
trans(::ConstantDLG) = (λ=asℝ₊, μ=asℝ₊, κ=asℝ₊, η=as𝕀)
Base.:*(m::ConstantDLG, x::Real) = ConstantDLG(λ=m.λ*x, μ=m.μ*x, κ=m.κ, η=m.η)
function (::ConstantDLG)(θ)
    t = promote_nt(θ)
    ConstantDLG(; λ=t.λ, μ=t.μ, κ=t.κ, η=t.η)
end


@with_kw struct ConstantDLGWGD{T} <: Params{T}
    λ::T
    μ::T
    q::Vector{T}
    κ::T = 0.
    η::T = 0.66
end

getθ(m::ConstantDLGWGD, node) = iswgd(node) ?
    (λ=m.λ, μ=m.μ, q=m.q[wgdid(node)], κ=m.κ) : (λ=m.λ, μ=m.μ, κ=m.κ, η=m.η)
trans(m::ConstantDLGWGD) = (
    λ=asℝ₊, μ=asℝ₊,
    q=as(Array, as𝕀, length(m.q)),
    κ=asℝ₊, η=as𝕀)
function (::ConstantDLGWGD)(θ)
    T = eltype(θ.q)
    ConstantDLGWGD(;λ=T(θ.λ), μ=T(θ.μ), q=θ.q, κ=T(θ.κ), η=T(θ.η))
end

"""
    DLG{T}

Simple branch-wise rates duplication-loss and gain model. Gain (κ) is
assumed to be tree-wide. This assumes a shifted geometric distribution
on the family size at the root with mean 1/η.
"""
@with_kw struct DLG{T} <: Params{T}
    λ::Vector{T}
    μ::Vector{T}
    κ::T = 0.
    η::T = 0.66
end

getθ(m::DLG, node) = (λ=exp(m.λ[id(node)]), μ=exp(m.μ[id(node)]), κ=m.κ, η=m.η)
trans(m::DLG) = (
    λ=as(Array, asℝ, length(m.λ)),
    μ=as(Array, asℝ, length(m.λ)),
    κ=asℝ₊, η=as𝕀)
(::DLG)(θ) = DLG(; λ=θ.λ, μ=θ.μ, κ=eltype(θ.λ)(θ.κ), η=eltype(θ.λ)(θ.η))
Base.:*(m::DLG, x::Real) = DLG(λ=m.λ.*x, μ=m.μ.*x, κ=m.κ, η=m.η)

@with_kw struct DLGWGD{T} <: Params{T}
    λ::Vector{T}
    μ::Vector{T}
    q::Vector{T}
    κ::T = 0.
    η::T = 0.66
end

function getθ(m::DLGWGD, node)
    return if iswgd(node)
        c = nonwgdchild(node)
        (λ=exp(m.λ[id(c)]), μ=exp(m.μ[id(c)]), q=m.q[wgdid(node)], κ=m.κ)
    else
        (λ=exp(m.λ[id(node)]), μ=exp(m.μ[id(node)]), κ=m.κ, η=m.η)
    end
end

trans(m::DLGWGD) = (
    λ=as(Array, asℝ, length(m.λ)),
    μ=as(Array, asℝ, length(m.λ)),
    q=as(Array, as𝕀, length(m.q)),
    κ=asℝ₊, η=as𝕀)

(::DLGWGD)(θ) = DLGWGD(;
    λ=θ.λ, μ=θ.μ, q=θ.q, κ=eltype(θ.λ)(θ.κ), η=eltype(θ.λ)(θ.η))

# short hands
ConstantDLWGD(; fixed=(:κ,), θ...) =
    RatesModel(ConstantDLGWGD(;θ...), fixed=mergetup(fixed, (:κ,)))
DLWGD(; fixed=(:κ,), θ...) =
    RatesModel(DLGWGD(;θ...), fixed=mergetup(fixed, (:κ,)))
mergetup(t1, t2) = tuple(union(t1, t2)...)

# Mixture wrapper; a marginalized mixture is natural to implement as a wrapper?
struct GammaMixture{M,T} <: Params{T}
    params::M
    rrates::Vector{T}
    α::T
    function GammaMixture(m::M, K; α=1.0) where M<:Params{T} where T
        qs = quantile.(Gamma(α,one(α)/α), collect((0+(1/2K)):(1/K):1))
        qs .*= K/sum(qs)
        new{M,T}(m, qs, T(α))
    end
end

trans(m::GammaMixture) = merge(trans(m.params), (α=asℝ₊,))
getθ(m::GammaMixture, node) = getθ(m.params, node)
(m::GammaMixture)(θ) = GammaMixture(m.params(θ), length(m.rrates), α=θ.α)
getparam(m::GammaMixture, v) = v != :α ? getparam(m.params, v) : m.α

# end
