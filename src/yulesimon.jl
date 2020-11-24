# Take this somewhere else...
module YuleSimonDistribution
using Distributions, SpecialFunctions 

struct YuleSimon{T} <: DiscreteUnivariateDistribution
    ρ::T
end

# Using the definition as an Exponential - Geometric mixed distribution
function Base.rand(d::YuleSimon)
    w = rand(Exponential(1/d.ρ))
    rand(Geometric(exp(-w))) + 1
end

function Base.rand(d::YuleSimon, n::Int64, ϵ=1e-10)
    ws = rand(Exponential(1/d.ρ), n)
    ps = exp.(-ws)
    ps[ps .>= 1.] .= 1. -ϵ
    rand.(Geometric.(ps)) .+ 1
end

Distributions.pdf(d::YuleSimon, k::Integer) = d.ρ*beta(k, d.ρ+1)
function Distributions.logpdf(d::YuleSimon, k::Integer) 
    try 
        log(d.ρ)+log(beta(k, d.ρ+1))
    catch e
        @warn e
        -Inf
    end
end

Distributions.mean(d::YuleSimon) = d.ρ > 1 ? d.ρ/(d.ρ-1.0) : NaN

function recursive_logpdf(d::YuleSimon, n)
    p = zeros(n)
    ρ′ = 1/d.ρ  # Yule's ρ = Simon's ρ⁻¹!
    p[1] = -log(1. + ρ′)
    for i=2:n
        p[i] = log(i-1) + log(ρ′) + p[i-1] - log(1.0 + i*ρ′)
    end
    p
end

export YuleSimon
end  # module
