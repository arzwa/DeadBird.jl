# defined here, but implemented elsewhere
function isawgm end
function wgmid end

function nonwgmchild(n)
   while isawgm(n)
       n = n[1]
   end
   return n
end

"""
    RatesModel

Abstract type for diferent rate models for phylogenies (e.g. constant rates
across the tree, branch-specific rates, models with WGD nodes, ...).
"""
abstract type RatesModel{T} end
const LinearModel{T} = RatesModel{T}  # currently non-linear models no longer supported

is_excessmodel(m::RatesModel) = false

#function Base.NamedTuple(m::M) where M<:RatesModel
#    return (; (k => getfield(m, k) for k in propertynames(m))...)
#end

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

ConstantDLG(λ, μ, κ) = ConstantDLG(promote(λ, μ, κ)...)

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
    excess = false
end

# promotion...
function ConstantDLGWGM(λ, μ, κ, q, e)
    λ, μ, κ = promote(λ, μ, κ)
    V = typeof(λ)
    q = Dict(i=>V(qi) for (i,qi) in q)
    ConstantDLGWGM(λ, μ, κ, q, e)
end

getθ(m::ConstantDLGWGM, node) = 
    (λ=m.λ, μ=m.μ, κ=m.κ, q=isawgm(node) ? m.q[wgmid(node)] : NaN)

is_excessmodel(m::ConstantDLGWGM) = m.excess

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
@with_kw struct DLGWGM{T,I} <: RatesModel{T}
    λ::Vector{T}
    μ::Vector{T}
    κ::Vector{T}
    q::Dict{I,T} = Dict{UInt16,Float64}()
    excess = false
end

# promotion...
function DLGWGM(λ, μ, κ, q, e)
    λ, μ, κ = promote(λ, μ, κ)
    V = eltype(λ)
    q = Dict(i=>V(qi) for (i,qi) in q)
    DLGWGM(λ, μ, κ, q, e)
end

function getθ(m::DLGWGM, node)
    return if isawgm(node)
        c = nonwgmchild(node)
        (λ=exp(m.λ[id(c)]), 
         μ=exp(m.μ[id(c)]), 
         κ=exp(m.κ[id(c)]), 
         q=m.q[wgmid(node)])
    else
        (λ=exp(m.λ[id(node)]), 
         μ=exp(m.μ[id(node)]), 
         κ=exp(m.κ[id(node)])) 
    end
end

is_excessmodel(m::DLGWGM) = m.excess
