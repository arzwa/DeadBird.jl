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
using Printf
using Random
import StatsBase: loglikelihood

#include("mexp.jl")
include("rmodels.jl")
include("model.jl")
include("countdag.jl")
include("profile.jl")
include("linear.jl")
#include("nonlinear.jl")
include("dlsim.jl")
include("ppsim.jl")
include("utils.jl")

"""
    example_data()

Get some example_data.

# Example (and benchmark)
```julia-repl
julia> x = DeadBird.example_data();

julia> @btime DeadBird.loglikelihood!(x.dag, x.model)
  36.974 μs (431 allocations: 31.53 KiB)
-26.30930561857625

julia> @btime DeadBird.loglikelihood!(x.mat, x.model)
  32.876 μs (420 allocations: 29.91 KiB)
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
export simulate

end # module
