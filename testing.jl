using NewickTree
using Parameters
using StatsFuns
using Test, BenchmarkTools

# ## 'Classical' approach
t = readnw(readline("example/9dicots.nw"))
r = ConstantDLG(λ=1.0, μ=1.2, η=0.9)
m = PhyloBDP(r, t, 13)
x = [12, 6, 1, 5, 2, 1, 1, 3, 1, 2, 6, 3, 1, 2, 3, 1, 2]
L = fill(-Inf, (13,17))
compute_conditionals!(L, x, m)

shouldbe = [-Inf, -9.756251, -11.236213, -18.206932, -26.424245, -35.508428,
            -45.336718, -55.847512, -67.019011, -78.884863, -91.455586,
            -104.807965, -119.2536]
@testset "Still OK?" begin
    for i=1:length(shouldbe) @test L[i,1] ≈ shouldbe[i] atol=1e-6 end
end
@btime cm!($L, $x, $m.nodes[4])
@btime compute_conditionals!($L, $x, $m)

t = readnw("((A:1,B:1):1,C:2);")
r = ConstantDLG(λ=1.0, μ=1.2, η=0.9)
x1 = [3, 2, 1, 1, 1]
L1 = fill(-Inf, (5,5))
x2 = [4, 2, 1, 1, 2]
L2 = fill(-Inf, (5,5))
x3 = [4, 2, 2, 0, 2]
L3 = fill(-Inf, (5,5))
m = PhyloBDP(r, t, 5)
compute_conditionals!(L1, x1, m)
compute_conditionals!(L2, x2, m)
compute_conditionals!(L3, x3, m);

# ## DAG approach
# Some tests for the DAG builder
using DelimitedFiles, NewickTree, Test, BenchmarkTools

X, s = readdlm("example/9dicots-f01-100.csv", ',', Int, header=true)
tree = readnw(readline("example/9dicots.nw"))
dag = builddag(X, s, tree)
g = dag.graph
@test outdegree(g, nv(g)) == length(unique(eachrow(X)))
@test sum([dag.ndata[i].count for i in outneighbors(g, nv(g))]) == size(X)[1]
# The graph contains 360 nodes, so I guess 360 calculations later. Naive calculation would result in 100 × 17 calculations I guess, and calculation of unique rows 81 × 17 = 1377. So we could expect some speed up

X, s = readdlm("example/9dicots-f01-1000.csv", ',', Int, header=true)
tree = readnw(readline("example/9dicots.nw"))
dag = builddag(X, s, tree)
g = dag.graph
@test outdegree(g, nv(g)) == length(unique(eachrow(X)))
@test sum([dag.ndata[i].count for i in outneighbors(g, nv(g))]) == size(X)[1]
# here we would go from 660 * 17 = 11220 to 1606

X, s = readdlm("example/9dicots-f01-25.csv", ',', Int, header=true)
tree = readnw(readline("example/9dicots.nw"))
dag = builddag(X, s, tree)
g = dag.graph
@test outdegree(g, nv(g)) == length(unique(eachrow(X)))
@test sum([dag.ndata[i].count for i in outneighbors(g, nv(g))]) == size(X)[1]

mmax = maximum([n.bound for n in dag.ndata])
r = ConstantDLG(λ=1.0, μ=1.2, κ=0.0 , η=0.9)
m = PhyloBDP(r, tree, 24+1)
@show likelihood!(dag, m)
@btime loglikelihood!(dag, m)

for n in outneighbors(dag.graph, nv(dag.graph))
    @show dag.ndata[n].count, round.(dag.parts[n][1:4], digits=2)
end

# previous version (25 gene families, 9 dicots)
# julia> @btime logpdf!(model, data)
#   1.953 ms (7648 allocations: 1.31 MiB)
# -251.0358357105553

# Now I get:
#   939.636 μs (3928 allocations: 380.89 KiB)
# But: -251.4429111574277

# (1000 gene families, 9 dicots)
# julia> @btime logpdf!(model, data)
#   85.427 ms (312904 allocations: 58.62 MiB)
# -12724.094725213423

# Now I get
#   28.533 ms (77725 allocations: 9.83 MiB)
# But: -12741.039240689497

# ## AD related
# This crashes
# using Zygote
# f = getgradclosure(dag, m)
# f(rand(2))
# gradient(x->f(x), rand(2))

using ForwardDiff, Optim
# X, s = readdlm("example/9dicots-f01-1000.csv", ',', Int, header=true)
X, s = readdlm("example/9dicots-f01-1000.csv", ',', Int, header=true)
tree = readnw(readline("example/9dicots.nw"))
dag, bound = CountDAG(X, s, tree)

r = RatesModel(ConstantDLG(λ=1.0, μ=1.0, κ=0.0 , η=1/1.5), fixed=(:η, :κ))
m = PhyloBDP(r, tree, bound)
loglikelihood!(dag, m)

problem = mle_problem(dag, m)
@time out = optimize(problem.f, problem.∇f, randn(2), BFGS())
m.rates.trans(out.minimizer)

nnodes = length(postwalk(tree))
r = RatesModel(DLG(λ=rand(nnodes), μ=rand(nnodes)), fixed=(:η, :κ))
m = PhyloBDP(r, tree, bound)
problem = mle_problem(dag, m)
out = optimize(problem.f, problem.∇f, randn(34), BFGS())
@info out
@show m.rates.trans(out.minimizer)


# HMC
using DynamicHMC, LogDensityProblems, Random, DynamicHMC.Diagnostics

# I actually don't mind having it like this, we subtype on problem, and code
# the prior in (::subproblem)(θ) function. We can code some logpdf functions
# for phylogenetic priors separately as building blocks.
struct ConstantDLGProblem <: Problem
    model
    data
end

function (p::ConstantDLGProblem)(θ)
    @unpack λ, μ, η, κ = θ
    ℓ = loglikelihood(p, θ)
    π = logpdf(MvLogNormal(ones(2)), [λ, μ]) +
         logpdf(Exponential(0.1), κ) +
         logpdf(Beta(1,3), η)
    return ℓ + π
end

r = RatesModel(ConstantDLG(λ=1.0, μ=1.2, κ=0.01 , η=1/mean(X)))
m = PhyloBDP(r, tree, bound)
p = ConstantDLGProblem(m, dag)
t = trans(p)
P = TransformedLogDensity(t, p)
∇P = ADgradient(:ForwardDiff, P);
results = mcmc_with_warmup(Random.GLOBAL_RNG, ∇P, 100);
posterior = transform.(t, results.chain)
@info summarize_tree_statistics(results.tree_statistics)
