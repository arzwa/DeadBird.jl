
using Pkg; Pkg.activate(joinpath(@__DIR__, "docs"))
using DeadBird
using Distributions, Turing, CSV, DataFrames, NewickTree, Optim, Parameters
using Plots, StatsPlots, Measures, StatsBase
default(legend=false, grid=false, titlefont=9, title_loc=:left)

tree = nw"((((A:1,B:1):1,C:2):1,D:3):1,E:4);"
plot(tree)
η = 0.75
λ = .1
μ = .1
p = ShiftedGeometric(η)

function mapssim(λ, μ, M)
    dfs = map([0.5, 1, 3]) do a
        M = M(rates=ConstantDLG(λ=a*λ, μ=a*μ, κ=0.))
        df = DeadBird.simulate(M, 1000)
        df = df[:,id.(getleaves(getroot(M)))]
    end
    df = vcat(dfs...)
    rename!(df, name.(getleaves(getroot(M))))
end

model1 = PhyloBDP(ConstantDLG(λ=λ, μ=μ, κ=0.), p, tree, 10)

@model constantrate0(M, dag, ::Type{T}=Float64) where T = begin
    λ ~ Exponential(1.5)
    μ ~ Exponential(1.5)
    dag ~ M(rates=ConstantDLG(λ=λ, μ=μ, κ=T(1e-10)))
end

using Random
Random.seed!(345)
xs = map(1:20) do rep
    @info rep
    λ = rand(LogNormal(log(0.1), 0.2))
    μ = exp(log(λ) + rand(Normal(0,0.1)))
    model1 = PhyloBDP(ConstantDLG(λ=λ, μ=μ, κ=0.), p, tree, 10)
    data = mapssim(λ, μ, model1)
    dag, bound = CountDAG(data, model1)
    model1 = model1(bound=bound)
    res1 = optimize(constantrate0(model1, dag), MLE())
    l, m = res1.values
    model2 = model1(rates=ConstantDLG(λ=l, μ=m, κ=0.))
    simdata = mapssim(λ, μ, model2)
    data, simdata, λ, μ, l, m
end

p1 = scatter(getindex.(xs, 3), getindex.(xs, 5), color=:black, label="\$\\lambda\$")
scatter!(getindex.(xs, 4), getindex.(xs, 6), color=:salmon, label="\$\\mu\$")
plot!(x->x, color=:black, size=(320,300), framestyle=:box, 
      fg_legend=:transparent,
      label="",
      legend=true,
      xlim=(0.07,0.17), 
      ylim=(0.07,0.17),
      xlabel="simulated",
      ylabel="MLE"
     )

savefig("/home/arzwa/research/maps/2022/doc/img/mles.pdf")


sizedist(df, is) = mapreduce(x->proportions(x, is), hcat, eachcol(df))
sizedist2(df, is) = proportions(vec(Matrix(df)), is)

ys = map(xs) do (d, s, _, _, _, _)
    a = sizedist(d, 0:2)
    b = sizedist(s, 0:2)
    plot(a[:,1]); plot!(b[:,1])
end
plot(ys...)


# Consider the estimated rates from the supplement of 1KP
# simulate data according to the approach of Li et al. (but what species tree?)
# compare statistics of the simulated trees with the 1KP data

# B4 analysis
λ, μ, η = 0.00701, 0.00585, 1.44

using DeadBird, Distributions, CSV, DataFrames, NewickTree
using Plots, StatsPlots, Measures, StatsBase
using Random

function profiletree(tree, leaves, delim="|")
    d = Dict(Symbol(x)=>0 for x in leaves)
    for k in getleaves(tree)
        x = Symbol(split(name(k), delim)[1])
        d[x] += 1
    end
    return (; d...)
end

# E43 analysis
# too bad I have no clue what species tree they have used...
tree = readnw(readline("/home/arzwa/research/maps/2022/data/caml-faa.tree"))
tree = readnw(readline("/home/arzwa/research/maps/2022/data/astral-binning-faa.tree"))
taxa = ["JNKW", "RFSD", "KGJF", "DOVJ", "CFRN", "HUQC", "IXVJ"]
tree = readnw(nwstr(NewickTree.extract(tree, taxa)))
for n in getleaves(tree)
    n.data.name = lowercase(n.data.name[1:3])
end
leaves = name.(getleaves(tree))
plot(tree)

Random.seed!(43)
λ, μ, η = 0.00704, 0.01417, 1/1.38
rp = ShiftedGeometric(η)
M  = PhyloBDP(ConstantDLG(λ=λ, μ=μ, κ=0.), rp, tree, 1, cond=:nowhere)
df = DeadBird.simulate(M, 1000)
df = DeadBird.observedmatrix(df, M)[:,1:length(taxa)] 

trees = map(readnw, readlines("/home/arzwa/research/maps/2022/data/E43/E43.maps.tre"))
X43 = DataFrame(map(t->profiletree(t, leaves), trees))

using Turing, Optim
@model constantrate0(M, dag, ::Type{T}=Float64) where T = begin
    λ ~ Exponential(1.5)
    μ ~ Exponential(1.5)
    dag ~ M(rates=ConstantDLG(λ=λ, μ=μ, κ=T(1e-10)))
end

dag, bound = CountDAG(X43, tree)
M2 = PhyloBDP(ConstantDLG(λ=λ, μ=μ, κ=0.), rp, tree, bound)
res = optimize(constantrate0(M2, dag), MLE())

