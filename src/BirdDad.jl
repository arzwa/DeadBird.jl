module BirdDad
# DeadBird, BirthdayDad, BirdHat, ...

using NewickTree
using Parameters
using StatsFuns
using Distributions
using LightGraphs
using StatsBase
using StatsFuns
using TransformVariables
using ForwardDiff
import StatsBase: loglikelihood

include("rmodels.jl")
include("model.jl")
include("countdag.jl")
include("inference.jl")

# TODO:
# - Discrete Gamma mixture
# - Dirichlet process mixture
# - Regression
# - Get WGDs in this implementation
# - Ordinary matrix based approach
# - Ancestral states https://gitlab.psb.ugent.be/arzwa/beluga/blob/reversible-jump/src/_em.jl

end # module
