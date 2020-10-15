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

"""
    example_data()

Get some example_data.

# Example
```julia-repl
julia> x = DeadBird.example_data();

julia> logpdf(x.model, x.dag)
-26.30930561857625

julia> logpdf(x.model, x.mat)
-26.309305618576246
```
"""
function example_data()
    tr = readnw("((A:1,B:1):0.5,C:1.5);")
    df = DataFrame(:A=>[1,1,3,0], :B=>[0,1,0,3], :C=>[1,1,4,4])
    dag, bound = CountDAG(df, tr)
    mat, bound = ProfileMatrix(df, tr)
    rates = RatesModel(ConstantDLG(λ=0.1,μ=0.1), fixed=(:κ,:η))
    model = PhyloBDP(rates, tr, bound)
    (tr=tr, df=df, dag=dag, mat=mat, model=model)
end

export CountDAG, ProfileMatrix, Profile
export ConstantDLG, DLG, RatesModel, PhyloBDP, ConstantDLSC, ModelArray
export simulate_profile, simulate_profile_ma

end # module
