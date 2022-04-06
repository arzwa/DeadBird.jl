# Load the required packages
using Pkg; Pkg.activate(joinpath(@__DIR__, "docs"))
using DeadBird
using Distributions, Turing, CSV, DataFrames, NewickTree, Optim
using Random; Random.seed!(761);
using ForwardDiff, Preferences
set_preferences!(ForwardDiff, "nansafe_mode" => true)

# Load the data
datadir = joinpath(@__DIR__, "docs/data")
tree = readnw(readline(joinpath(datadir, "primates-11taxa.nw")))
data = CSV.read(joinpath(datadir, "primates-GO0002376-oib-max10.csv"), DataFrame);

taxa = name.(getleaves(readnw(readline(joinpath(datadir, "primates-11taxa-fullname.nw")))))
taxa = Dict(zip(name.(getleaves(tree)), taxa))

# The data set size and number of taxa are
nrow(data), length(getleaves(tree))

# The average number of genes in non-extinct families is
m = mean(filter(x->x>0,Matrix(data)))

# We can use this to parameterize the prior for the number of ancestral
# lineages
η = 0.92
ζ = 3.23
rootprior = ShiftedBetaGeometric(η, ζ)
#rootprior = ShiftedGeometric(η)

# We will use the DAG data structure (most efficient, but admits no
# family-specific models).
dag, bound = CountDAG(data, tree)

# We will define a Turing model for this simple problem
@model constantrate(dag, bound, tree, rootprior, ::Type{T}=Float64) where T = begin
    λ ~ Exponential()
    μ ~ Exponential()
    θ = ConstantDLG(λ=λ, μ=μ, κ=T(1e-10))
    dag ~ PhyloBDP(θ, rootprior, tree, bound)
end

chn = sample(constantrate(dag, bound, tree, rootprior), NUTS(), 200)

# Posterior predictive check
M = PhyloBDP(ConstantDLG(rand(3)...), rootprior, tree, bound)
function mfun(M, x)
    l, m = Array(x)
    M(rates=ConstantDLG(λ=l, μ=m, κ=0.))
end
pp = DeadBird.simulate(y->mfun(M, y), data, chn, 1000)


# Discrete mixture
@model mixturerate(dag, bound, tree, rootprior, K=4, ::Type{T}=Float64) where T = begin
    λ ~ Exponential()
    μ ~ Exponential()
    σ ~ Exponential()
    m = -σ^2/2  # mean 1  (mean = exp(m + σ^2/2) with m, σ Normal params)
    a = DeadBird.discretize(LogNormal(m, σ), K)
    θ = [ConstantDLG(λ=λ * a[i], μ=μ * a[i], κ=T(1e-10)) for i=1:K]
    mix = [PhyloBDP(θi, rootprior, tree, bound) for θi in θ]
    dag ~ MixtureModel(mix, fill(1/K, K))
end

dag, bound = CountDAG(data, tree)
K = 8
chn2 = sample(mixturerate(dag, bound, tree, rootprior, K), NUTS(), 400)

s = mean(chn2[:σ])
d = LogNormal(-s^2/2, s)
a = DeadBird.discretize(d, K)
plot(d, grid=false, legend=false, color=:lightgray, fill=true, xlim=(0,1.5a[end]), ylim=(0,Inf)); 
vline!(a, color=:black)
vline!([1], linestyle=:dot, color=:black)
    
M = PhyloBDP(ConstantDLG(rand(3)...), rootprior, tree, bound)
function mfun2(M, K, x)
    l = x[:λ][1]
    m = x[:μ][1]
    σ = x[:σ][1]
    a = rand(DeadBird.discretize(LogNormal(-σ^2/2, σ), K))
    M(rates=ConstantDLG(λ=a*l, μ=a*m, κ=zero(l)))
end
pp2 = DeadBird.simulate(y->mfun2(M, K, y), data, chn2, 1000)


p2 = plot(pp2, size=(700,500), color=:gray, xscale=:identity, ylim=(-Inf, 0.5), xlim=(1,12),
     taxa=taxa, titlefontfamily="helvetica oblique", titlefont=12,
     xticks=(1.5:5:11.5, 0:5:10))

p1 = plot!(p2, pp, color=:salmon, size=(700,500), xscale=:identity, ylim=(-Inf, 0.5), xlim=(1,12),
     taxa=taxa, titlefontfamily="helvetica oblique", titlefont=12,
     xticks=(1.5:5:11.5, 0:5:10))

savefig("docs/img/primates-crK8-pps.pdf")
