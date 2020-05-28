module BirdDad
# DeadBird, BirthdayDad, BirdHat, ...

using NewickTree
using Parameters
using StatsFuns
using Distributions
using LightGraphs
using StatsBase
using StatsFuns
using FakeFamily
using TransformVariables
using ForwardDiff
import StatsBase: loglikelihood
import FakeFamily: Params, RatesModel, getθ, trans

include("rmodels.jl")
include("model.jl")
include("countdag.jl")
include("inference.jl")

# TODO:
# - Discrete Gamma mixture
# - Dirichlet process mixture
# - Regression
# - Ordinary matrix based approach
# - Ancestral states https://gitlab.psb.ugent.be/arzwa/beluga/blob/reversible-jump/src/_em.jl

end # module
