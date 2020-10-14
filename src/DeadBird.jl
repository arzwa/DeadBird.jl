module DeadBird

using NewickTree
using Parameters
using Distributions
using LightGraphs
using StatsBase
using StatsFuns
using TransformVariables
using DataFrames
using BandedMatrices
using ThreadTools
using RecipesBase
import StatsBase: loglikelihood

include("mexp.jl")
include("rmodels.jl")
include("model.jl")
include("countdag.jl")
include("profile.jl")
include("linear.jl")
include("nonlinear.jl")
include("dlsim.jl")
include("ppsim.jl")
include("utils.jl")

export CountDAG, ProfileMatrix, Profile
export ConstantDLG, DLG, RatesModel, PhyloBDP, ConstantDLSC, ModelArray
export simulate_profile, simulate_profile_ma

end # module
