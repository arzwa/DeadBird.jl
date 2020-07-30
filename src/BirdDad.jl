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
using DataFrames
using BandedMatrices
import StatsBase: loglikelihood
# import FakeFamily: Params, RatesModel, getθ, trans

include("rmodels.jl")
include("model.jl")
include("countdag.jl")
include("profile.jl")
include("dlsim.jl")
include("ppsim.jl")

include("truncmodel.jl")

export CountDAG, ProfileMatrix, Profile
export ConstantDLG, DLG, RatesModel, PhyloBDP, GammaMixture
export mle_problem

# TODO:
# - Discrete Gamma mixture
# - Dirichlet process mixture
# - Regression
# - Ordinary matrix based approach
# - Ancestral states https://gitlab.psb.ugent.be/arzwa/beluga/blob/reversible-jump/src/_em.jl

end # module
