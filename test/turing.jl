#using Pkg; Pkg.activate(@__DIR__)
using Test, NewickTree, BirdDad, Turing
using DelimitedFiles, Random, Distributions, CSV

Random.seed!(624)
const datadir = joinpath(@__DIR__, "example")
df = CSV.read(joinpath(datadir, "9dicots-f01-100.csv"))
tree = readnw(readline(joinpath(datadir, "9dicots.nw")))


# CountDAG
# Construct the data object
dag, bound = CountDAG(df, tree)

# Construct the model
rates = RatesModel(ConstantDLG(λ=.1, μ=.1, κ=0.0, η=1/1.5), fixed=(:κ,))
model = PhyloBDP(rates, tree, bound)

@model constantrates(dag, model) = begin
    r ~ MvLogNormal(ones(2))
    η ~ Beta(2,1)
    dag ~ model((λ=r[1], μ=r[2], η=η))
end

bmodel = constantrates(dag, model)
chain = sample(bmodel, NUTS(0.65), 1000);


# ProfileMatrix
matrix, bound = BirdDad.ProfileMatrix(df, tree)

@model familyrates(matrix, model, ::Type{T}=Float64) where T = begin
    n = size(matrix)[1]
    η ~ Beta(2,1)
    α ~ Exponential()
    r ~ Exponential()
    rr = Vector{T}(undef, n)
    for i=1:n
        rr[i] ~ Gamma(α, 1/α)
        matrix[i] ~ model((λ=rr[i]*r, μ=rr[i]*r, η=η))
    end
end

bmodel = familyrates(matrix[1:10], model)
chain = sample(bmodel, NUTS(0.65), 200);


# Gain model for core families
core = df[filter(i->all(Array(df[i,:]) .> 0), 1:size(df)[1]), :]
dag, bound = BirdDad.CountDAG(core .- 1, tree)

rates = RatesModel(ConstantDLG(λ=.1, μ=.13, κ=.1, η=1/1.5), 
                   fixed=(), rootprior=:geometric)
model = PhyloBDP(rates, tree, bound, cond=:none)

@model corefams(dag, model) = begin
    λ ~ Exponential()
    α ~ Exponential()
    μ = λ*exp(α)
    dag ~ model((λ=λ, μ=μ, κ=λ, η=1.0-λ/μ))
end

bmodel = corefams(dag, model)
chain = sample(bmodel, NUTS(0.65), 200);
