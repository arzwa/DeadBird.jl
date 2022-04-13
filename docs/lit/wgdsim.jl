using Pkg; Pkg.activate(joinpath(@__DIR__, "docs"))
using DeadBird
using Distributions, Turing, CSV, DataFrames, NewickTree, Optim, Parameters
using Plots, StatsPlots, Measures, StatsBase

dir = "docs/data/dicots"
tree = readnw(readline(joinpath(dir, "9dicots.nw")))
n = postwalk(tree)[4]
taxa = Dict("vvi"=>"V. vinifera", 
            "ath"=>"A. thaliana", 
            "cpa"=>"C. papaya", 
            "mtr"=>"M. truncatula", 
            "ptr"=>"P. trichocarpa", 
            "bvu"=>"B. vulgaris", 
            "cqu"=>"C. quinoa", 
            "ugi"=>"U. gibba", 
            "sly"=>"S. lycopersicum")

λ = μ = 1.5
η = 0.75
p = ShiftedGeometric(η)
M = PhyloBDP(ConstantDLG(λ=λ, μ=μ, κ=0.), p, tree, 10)

df = DeadBird.simulate(M, 1000)
dag, bound = CountDAG(df, tree)
M = M(bound=bound)

@model constantrate(M, dag, ::Type{T}=Float64) where T = begin
    λ ~ Exponential(1.5)
    μ ~ Exponential(1.5)
    dag ~ M(rates=ConstantDLG(λ=λ, μ=μ, κ=T(1e-10)))
end

chn1 = sample(constantrate(M, dag), NUTS(), 200)

M = PhyloBDP(ConstantDLGWGM(λ=λ, μ=μ, κ=0.), p, tree, 10)
M = DeadBird.insertwgms(M, id(n)=>(distance(n)/2, 2, 0.15))

df = DeadBird.simulate(M, 1000)
dag, bound = CountDAG(df, tree)
M = M(bound=bound)

@model constantrate2(M, dag, ::Type{T}=Float64) where T = begin
    λ ~ Exponential(1.5)
    μ ~ Exponential(1.5)
    q ~ Beta()
    dag ~ M(rates=ConstantDLGWGM(λ=λ, μ=μ, κ=T(1e-10), q=Dict(0x0012=>q)))
end

chn2 = sample(constantrate2(M, dag), NUTS(), 200)


