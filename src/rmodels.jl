# defined here, but implemented elsewhere
function isawgm end
function wgmid end
function nonwgmchild end

"""
    RatesModel

Abstract type for diferent rate models for phylogenies (e.g. constant rates
across the tree, branch-specific rates, models with WGD nodes, ...).
"""
abstract type RatesModel{T} end
const LinearModel{T} = RatesModel{T}  # currently non-linear models no longer supported

function Base.NamedTuple(m::M) where M<:RatesModel
    return (; (k => getfield(m, k) for k in propertynames(m))...)
end

"""
    newmodel(m::M, θ) where M<:RatesModel

Construct a new model of type `M` by taking the parameters of `m` and
parameters defined in the named tuple `θ`, the latter overriding the former.
"""
newmodel(m::M, θ) where M<:RatesModel = M(merge(NamedTuple(m), θ)...)

"""
    getθ(m<:RatesModel, node)

Get the parameter values from a `RatesModel` relevant for a particular node in
a phylogeny. Should be implemented for each RatesModel where parameters differ
across branches.
"""
getθ(m::M, node) where M<:RatesModel = m  # default

"""
    ConstantDLG{T}

Simple constant rates duplication-loss and gain model. All nodes of the tree
are associated with the same parameters (duplication rate λ, loss rate μ, gain
rate κ). 
"""
@with_kw struct ConstantDLG{T} <: RatesModel{T}
    λ::T = 0.1
    μ::T = 0.1
    κ::T = 0.
end

"""
    ConstantDLGWGM{T}

Similar to `ConstantDLG`, but with a field for whole-genome multiplication
(WGM) nodes in the phylogeny, which have a single retention rate parameter
`q` each.
"""
@with_kw struct ConstantDLGWGM{T,I} <: RatesModel{T}
    λ::T = 0.3
    μ::T = 0.5
    κ::T = 0.
    q::Dict{I,T} = Dict{UInt16,Float64}()
end

getθ(m::ConstantDLGWGM, node) = 
    (λ=m.λ, μ=m.μ, κ=m.κ, q=isawgm(node) ? m.q[wgmid(node)] : nothing)

"""
    DLG{T}

Simple branch-wise rates duplication-loss and gain model.  
"""
@with_kw struct DLG{T} <: RatesModel{T}
    λ::Vector{T}
    μ::Vector{T}
    κ::Vector{T}
end

DLG(λ, μ, κ) = DLG(promote(λ, μ, κ)...)

# XXX: why exactly do we keep them on log-scale?
getθ(m::DLG, node) = (
    λ=exp(m.λ[id(node)]), 
    μ=exp(m.μ[id(node)]), 
    κ=exp(m.κ[id(node)])) 

"""
    DLGWGM{T}

Similar to `DLG`, but with WGM nodes, see also `ConstantDLGWGM`.
"""
@with_kw struct DLGWGM{T} <: RatesModel{T}
    λ::Vector{T}
    μ::Vector{T}
    q::Vector{T}  # make it a dict or think of something better altogether
    κ::Vector{T}
end

DLG(λ, μ, q, κ) = DLG(promote(λ, μ, q, κ)...)

function getθ(m::DLGWGM, node)
    return if isawgm(node)
        c = nonwgmchild(node)
        (λ=exp(m.λ[id(c)]), μ=exp(m.μ[id(c)]), κ=exp(m.κ[id(c)]), q=m.q[wgmid(node)])
    else
        (λ=exp(m.λ[id(node)]), μ=exp(m.μ[id(node)]), κ=exp(m.κ[id(node)]), η=m.η)
    end
end
