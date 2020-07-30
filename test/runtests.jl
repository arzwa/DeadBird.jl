using Pkg; Pkg.activate(@__DIR__)
using Test, LightGraphs, NewickTree, BirdDad, TransformVariables
using DelimitedFiles, Random
Random.seed!(624)

# NOTE still needs NaN-safe mode enabled sometimes (when κ = 0 for instance)...

const ML = false

@testset "BirdDad tests" begin
    include("model.jl")
    ML && include("mle.jl")
end
