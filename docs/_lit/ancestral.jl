# verify correctness of ancestral state sampler

using Pkg; Pkg.activate(joinpath(@__DIR__, "docs"))
using DeadBird
using Distributions, Turing, CSV, DataFrames, NewickTree, Optim, Parameters
using Plots, StatsPlots, Measures, StatsBase

# No WGD
dir = "docs/data/dicots"
tree = readnw(readline(joinpath(dir, "9dicots.nw")))
node = postwalk(tree)[4]
λ = μ = 1.5
η = 0.75
p = ShiftedGeometric(η)

@model constantrate0(M, dag, ::Type{T}=Float64) where T = begin
    λ ~ Exponential(1.5)
    μ ~ Exponential(1.5)
    dag ~ M(rates=ConstantDLG(λ=λ, μ=μ, κ=T(1e-10)))
end

M = PhyloBDP(ConstantDLG(λ=λ, μ=μ, κ=1e-10), p, tree, 1)
df = DeadBird.simulate(M, 1000)
df_ = DeadBird.observedmatrix(df, M)
df_ = df_[:,1:9]

dag, bound = CountDAG(df_, tree)
M = PhyloBDP(ConstantDLG(λ=λ, μ=μ, κ=1e-10), p, tree, bound)
chn = sample(constantrate0(M, dag), NUTS(), 100)

function mfun(model, x)
    λ = get(x, :λ).λ[1]
    μ = get(x, :μ).μ[1]
    model(rates=ConstantDLG(λ=λ, μ=μ, κ=1e-10))
end

mat, _ = ProfileMatrix(df_, tree)
Xs = DeadBird.sample_ancestral(x->mfun(M, x), chn, mat, 100)

Ys = map(i->DeadBird.simulate(mfun(M, chn[i]), 1000), 1:length(chn))
Ys = cat(map(y->permutedims(Matrix(y[:,1:17])), Ys)..., dims=3)

ps = map(1:7) do i
    zz = proportionmap(data[:,i])
    xx = proportionmap(vec(Xs[i,:,:]))
    yy = proportionmap(vec(Ys[i,:,:]))
    p = bar(xx, color=:white)
    bar!(yy, color=:salmon, alpha=0.3)
    scatter!(zz)
end
plot(ps..., size=(900,700))

A = DeadBird.AncestralSampler(M, bound)
DeadBird.sample_ancestral(A, mat[1])


# With WGD
tree = readnw(readline(joinpath(dir, "9dicots.nw")))
node = postwalk(tree)[4]
λ = μ = 1.0
η = 0.75
p = ShiftedGeometric(η)
ns = [getlca(tree, "ptr"), getlca(tree, "cqu"),]
wgds = [id(n)=>(distance(n)/2, 2, 0.5) for n in ns]


@model constantrate1(M, dag, ::Type{T}=Float64) where T = begin
    λ ~ Exponential()
    μ ~ Exponential()
    q1 ~ Beta()
    q2 ~ Beta()
    qd = Dict(0x0012=>q1, 0x0014=>q2)
    dag ~ M(rates=ConstantDLGWGM(λ=λ, μ=μ, κ=T(1e-10), q=qd))
end

M = PhyloBDP(ConstantDLGWGM(λ=λ, μ=μ, κ=1e-10), p, tree, 5)
M = DeadBird.insertwgms(M, wgds...)
df = DeadBird.simulate(M, 1000)
df_ = DeadBird.observedmatrix(df, M)
df_ = df_[:,1:9]

dag, bound = CountDAG(df_, getroot(M))
M = M(bound=bound)
chn = sample(constantrate1(M, dag), NUTS(), 100)

function mfun(model, x)
    λ = get(x, :λ).λ[1]
    μ = get(x, :μ).μ[1]
    q1 = get(x, :q1).q1[1]
    q2 = get(x, :q2).q2[1]
    qd = Dict(0x0012=>q1, 0x0014=>q2)
    model(rates=ConstantDLGWGM(λ=λ, μ=μ, κ=1e-10, q=qd))
end

mat, _ = ProfileMatrix(df_, getroot(M))
Xs = DeadBird.sample_ancestral(x->mfun(M, x), chn, mat, 100)

Ys = map(i->DeadBird.simulate(mfun(M, chn[i]), 1000), 1:length(chn))
Ys = cat(map(y->permutedims(Matrix(y[:,1:length(M)])), Ys)..., dims=3)

ps = map(1:21) do i
    zz = proportionmap(df[:,i])
    xx = proportionmap(vec(Xs[i,:,:]))
    yy = proportionmap(vec(Ys[i,:,:]))
    p = bar(xx, color=:white, legend=false, grid=false)
    bar!(yy, color=:salmon, alpha=0.3)
    scatter!(zz)
end
plot(ps..., size=(900,700))

A = DeadBird.AncestralSampler(M, bound)
DeadBird.sample_ancestral(A, mat[1])
