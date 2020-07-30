function iswgd end
function wgdid end
function nonwgdchild end

abstract type Params{T} end

struct RatesModel{T,M<:Params{T},V}
    params::M
    fixed ::Tuple
    trans ::V
end

RatesModel(θ; fixed=()) = RatesModel(θ, fixed, gettrans(θ, fixed))

Base.eltype(m::RatesModel{T}) where T = T
Base.show(io::IO, m::RatesModel) = write(io,
    "RatesModel with $(m.fixed) fixed\n$(m.params)")

getθ(m::RatesModel, node) = getθ(m.params, node)
getp(m::P, n) where {T,P<:Params{T}} = hasfield(P, :p) &&
    length(m.p) > 0 && isleaf(n) ? m.p[id(n)] : 0.

# HACK: a little bit of metaprogramming to allow fixed parameters, necessary?
function gettrans(p::P, fixed) where P<:Params
    inner = join(["$k=$v," for (k,v) in pairs(trans(p)) if k ∉ fixed])
    expr  = Meta.parse("as(($inner))")
    eval(expr)
end

(m::RatesModel)(x::Vector) = m(m.trans(x))
function (m::RatesModel)(θ)
    θ′ = merge(θ, [k=>findfield(m.params, k) for k in m.fixed])
    RatesModel(m.params(θ′), m.fixed, m.trans)
end

function findfield(p::P, f) where {P<:Params{T} where T}
    hasfield(P, f) ? getfield(p, f) : findfield(p.params, f)
end

Base.rand(m::M) where M<:RatesModel = m(m.trans(randn(dimension(m.trans))))

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

promote_nt(nt) = (;zip(keys(nt), promote(nt...))...)

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
    q::Vector{T} = Float64[]
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

const LinearModel = RatesModel{T,V} where
    {T,V<:Union{ConstantDLG,DLG,DLGWGD,ConstantDLGWGD}}
