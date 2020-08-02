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
using DataFrames
using BandedMatrices
import StatsBase: loglikelihood
# import FakeFamily: Params, RatesModel, getθ, trans

include("mexp.jl")
include("rmodels.jl")
include("model.jl")
include("countdag.jl")
include("profile.jl")
include("linear.jl")
include("nonlinear.jl")
include("dlsim.jl")
include("ppsim.jl")

export CountDAG, ProfileMatrix, Profile
export ConstantDLG, DLG, RatesModel, PhyloBDP, ConstantDLSC

# TODO:
# - WGDs in matrix based approach
# - Dirichlet process mixture
# - Regression
# - Ancestral states https://gitlab.psb.ugent.be/arzwa/beluga/blob/reversible-jump/src/_em.jl

end # module
