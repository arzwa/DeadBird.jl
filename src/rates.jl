# The RatesModel provides an interfce between parameter values and a general
# phylogenetic model, so that we can use the same algorithm routines
# irrespective of how parameters are shared across nodes/branches/families,...
# IDEA: define a WGD model as a wrapper around a RatesModel.
# NOTE: the ratesmodel should also contain hyperparameters, and priors should
# be defined in terms of RatesModels...
# XXX: should we have Gamma mixtures baked in? Or should that be another
# wrapper around the ratesmodel?

abstract type Params{T} end

struct RatesModel{T,M<:Params{T},V}
    params::M
    fixed ::Tuple
    trans ::V
end

Base.eltype(m::RatesModel{T}) where T = T
RatesModel(θ; fixed=(:K,)) = RatesModel(θ, fixed, gettrans(θ, fixed))
Base.show(io::IO, m::RatesModel) = write(io,
    "RatesModel with $(m.fixed) fixed\n$(m.params)")
getθ(m::RatesModel, node) = getθ(m.params, node)

# HACK: a little bit of metaprogramming to allow fixed parameters, necessary?
function gettrans(p::P, fixed) where P<:Params
    inner = join(["$k=$v," for (k,v) in pairs(trans(p)) if k ∉ fixed])
    expr  = Meta.parse("as(($inner))")
    eval(expr)
end

function (m::RatesModel)(θ)
    θ′ = merge(θ, [k=>getfield(m.params, k) for k in m.fixed])
    RatesModel(m.params(θ′), m.fixed, m.trans)
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
    # α::T = 1.
    # K::Int = 1
end

getθ(m::ConstantDLG, node) = m
trans(::ConstantDLG) = (λ=asℝ₊, μ=asℝ₊, κ=asℝ₊, η=as𝕀)
# trans(::ConstantDLG) = (λ=asℝ₊, μ=asℝ₊, κ=asℝ₊, η=as𝕀, α=asℝ₊)
# the zip is a bit slow...
(::ConstantDLG)(θ) = ConstantDLG(; zip(keys(θ), promote(θ...))...)

@with_kw struct DLG{T} <: Params{T}
    λ::Vector{T}
    μ::Vector{T}
    κ::T = 0.
    η::T = 0.66
end

getθ(m::DLG, node) = (λ=m.λ[id(node)], μ=m.μ[id(node)], κ=m.κ, η=m.η)
trans(m::DLG) = (λ=as(Array, asℝ₊, length(m.λ)),
    μ=as(Array, asℝ₊, length(m.λ)), κ=asℝ₊, η=as𝕀)
(::DLG)(θ) = DLG(; λ=θ.λ, μ=θ.μ, κ=eltype(θ.λ)(θ.κ), η=eltype(θ.λ)(θ.η))
