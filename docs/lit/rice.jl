using Pkg; Pkg.activate(joinpath(@__DIR__, "docs"))
using DeadBird
using Distributions, Turing, CSV, DataFrames, NewickTree, Optim, Parameters
using Plots, StatsPlots, Measures
using StatsFuns, SpecialFunctions

dir = "/home/arzwa/research/gene-family-evolution/data/oryzinae"
data = CSV.read("$dir/oryza-max10-6taxa-oib.csv", DataFrame)
tree = readnw(readline(joinpath(dir, "oryzinae.6taxa.nw")))

#dataf = CSV.read("$dir/oryzinae.N0.tsv", DataFrame)
#X, cols = DeadBird.getoib(dataf, tree)
#data = X[:,cols]

taxa = Dict(x=>x[1][1] * ". " * join(split(x, "_")[2:end], " ") for x in name.(getleaves(tree)))

# get a named tuple for a chain 'row'
function getparams(x)
    vars = x.value.axes[2]
    (; [var=>x for (var,x) in zip(vars, vec(x.value.data))]...)
end

# We will use the DAG data structure (most efficient, but admits no
# family-specific models).
dag, bound = CountDAG(data, tree)

# We will define a Turing model for this simple problem
@model constantrate(dag, bound, tree, ::Type{T}=Float64) where T = begin
    η ~ Beta()
    ζ ~ Turing.FlatPos(1.)
    λ ~ Turing.FlatPos(0.)
    μ ~ Turing.FlatPos(0.)
    θ = ConstantDLG(λ=λ, μ=μ, κ=T(1e-10))
    p = ShiftedBetaGeometric(η, ζ)
    dag ~ PhyloBDP(θ, p, tree, bound)
end

chn1 = sample(constantrate(dag, bound, tree), NUTS(), 500)

function mfun1(x, tree, bound)
    @unpack λ, μ, η, ζ = getparams(x)
    PhyloBDP(ConstantDLG(λ=λ, μ=μ, κ=0.), ShiftedBetaGeometric(η, ζ), tree, bound)
end
pp1 = DeadBird.simulate(y->mfun1(y, tree, bound), data, chn1, 1000)

p1 = plot(pp1, xscale=:identity, xlim=(1,11), xticks=(1:11, 0:10))

# compare with paranome estimate
@model bgfit(xs) = begin
    η ~ Beta()
    ζ ~ Turing.FlatPos(0.)
    M = ShiftedBetaGeometric(η, ζ)
    for (k,x) in enumerate(xs)
        Turing.@addlogprob!(x*logpdf(M,k))
    end
end

bgfits = map(1:6) do i
    xs = counts(filter(x->x > 0, data[:,i]))
    chn = sample(bgfit(xs), NUTS(), 200)
end

# BDIP model for excess genes
nonextinct = filter(x->all(Array(x) .> 0), data);

# We will model the excess number of genes, i.e. the number of extra
# (duplicated) genes *per* family, instead of the total number of genes. 
excessgenes = nonextinct .- 1;

# Again we construct a DAG object
dag, bound = CountDAG(excessgenes, tree)

# The model we specify is a linear birth-death and immigration process with
# immigration (gain) rate equal to the duplication rate, `κ = λ`, and loss rate
# `μ`. This corresponds to a model where genes duplicate at rate λ, (note that
# a `0 -> 1` transition is also a duplication here since the zero state
# corresponds to a single copy family), and where *duplicated genes* get lost
# at rate `μ`. We assume `λ < μ`, in which case there is a geometric stationary
# distribution with mean `1 - λ/μ` for the excess number of genes in a family.
bound01(η) = η <= zero(η) ? zero(η) + 1e-16 : η >= one(η) ? one(η) - 1e-16 : η

@model nonextinctmodel(dag, bound, tree) = begin
    μ ~ Turing.FlatPos(0.)
    η ~ Beta(1, 1)  # 1 - λ/μ
    ζ ~ Exponential(10.)
    η = bound01(η)  
    λ = μ * (1 - η)
    rates = ConstantDLG(λ=λ, μ=μ, κ=λ)
    rootp = BetaGeometric(η, 1 + ζ)
    dag ~ PhyloBDP(rates, rootp, tree, bound, cond=:none)
end

# and we sample
chn2 = sample(nonextinctmodel(dag, bound, tree), NUTS(), 500)

function mfun2(x, tree, bound) 
    @unpack η, ζ, μ = getparams(x)
    λ = μ * (1 - η)
    rootp = BetaGeometric(η, 1 + ζ)
    M = PhyloBDP(ConstantDLG(λ=λ, μ=μ, κ=λ), rootp, tree, bound, cond=:none)
end
pp2 = DeadBird.simulate(y->mfun2(y, tree, bound), excessgenes, chn2, 1000)

p1 = plot(pp1, taxa=taxa, xscale=:identity, xlim=(1,11), xticks=(1:11, 0:10), xlabel="", ylabel="")
p2 = plot(pp2, taxa=taxa, color=:salmon, xscale=:identity, xlim=(1,11), xticks=(1:11, 0:10), xlabel="", ylabel="")
for p in [p1, p2]
    xlabel!.(p.subplots[4:end], "\$n\$")
    for sp in p.subplots[4:end]
        sp.attr[:bottom_margin] = 5mm
    end
end
for sp in p1.subplots[[1,4]]
    ylabel!(sp, "\$\\log_{10}f_n\$")
    sp.attr[:left_margin] = 5mm 
end
plot(p1, p2, size=(1000,250), titlefont=8)

savefig("docs/img/rice-constant-rates.pdf")

# Normal approximatoin to beta, and discretize
expit(x) = exp(x)/(1+exp(x))
function discretebeta(η, ζ, K)
    α = η*ζ
    β = (1-η)*ζ
    m = digamma(α) - digamma(β)
    v = trigamma(α) + trigamma(β)
    ps = expit.(DeadBird.discretize(Normal(m, √v), K))
end

@model nonextinctmodelmix(dag, bound, tree, K=4) = begin
    η ~ Beta(1, 1)  # 1 - λ/μ
    ζ ~ Exponential(10.)
    μ ~ Turing.FlatPos(0.)
    η = bound01(η)  
    α = discretebeta(η, 1+ζ, K)
    λ = μ .* (1 .- α)
    θ = [ConstantDLG(λ=λ[i], μ=μ, κ=λ[i]) for i=1:K]
    p = BetaGeometric(η, 1+ζ)
    dag ~ MixtureModel([PhyloBDP(θ[i], p, tree, bound, cond=:none) for i=1:K])
end

K = 4
dag, bound = CountDAG(excessgenes, tree)
chn3 = sample(nonextinctmodelmix(dag, bound, tree, K), NUTS(), 500)

function mfun3(x, tree, bound, K) 
    @unpack η, ζ, μ = getparams(x)
    λ = μ .* (1 .- normalapproxdiscrete(η, 1+ζ, K))
    rates = [ConstantDLG(λ=λ[i], μ=μ, κ=λ[i]) for i=1:K]
    rootp = BetaGeometric(η, 1+ζ)
    M = MixtureModel([PhyloBDP(rates[i], rootp, tree, bound, cond=:none) for i=1:K])
end
pp3 = DeadBird.simulate(y->mfun3(y, tree, bound, K), excessgenes, chn3, 1000)

p2 = plot(pp2, taxa=taxa, xscale=:identity, layout=(1,6), size=(1000,150), ylabel="")
plot!(pp3, taxa=taxa, color=:salmon, xscale=:identity, xlim=(1,11), xticks=(1:11, 0:10), ylabel="", bottom_margin=8mm)
p2.subplots[1].attr[:left_margin] = 5mm
ylabel!(p2.subplots[1], "\$\\log_{10}f_n\$")
plot(p2)
savefig("docs/img/rice-mix.pdf")
