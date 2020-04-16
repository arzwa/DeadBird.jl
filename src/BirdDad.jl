module BirdDad

using NewickTree
using Parameters
using StatsFuns
using Distributions
using LightGraphs
using StatsBase
using StatsFuns
using TransformVariables
import StatsBase: loglikelihood

include("rates.jl")
include("model.jl")
include("countdag.jl")
include("inference.jl")

# TODO:
# - Discrete Gamma mixture
# - Parallel computing for the DAG based approach
# - Explore Turing-based inference
# - Dirichlet process mixture
# - Regression
# - Get WGDs in this implementation
# - Ordinary matrix based approach

end # module
