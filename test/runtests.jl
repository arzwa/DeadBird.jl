using Pkg; Pkg.activate(@__DIR__)
using Test, LightGraphs, NewickTree, DeadBird, TransformVariables
using DelimitedFiles, Random
Random.seed!(624)

# NOTE still needs NaN-safe mode enabled sometimes (when κ = 0 for instance)...

const TURING = false

@testset "DeadBird tests" begin
    include("linear.jl")
end
