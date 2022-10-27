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

ftree = readnw(readline(joinpath(datadir, "primates-11taxa-fullname.nw")))
taxa = name.(getleaves(ftree))
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


p2 = plot(pp2, size=(700,500), color=:gray, xscale=:identity, ylim=(-Inf, 0.5),
          xlim=(1,12), taxa=taxa, titlefontfamily="helvetica oblique",
          titlefont=12, xticks=(1.5:5:11.5, 0:5:10))

p1 = plot!(p2, pp, color=:salmon, size=(700,500), xscale=:identity, 
           ylim=(-Inf, 0.5), xlim=(1,12), taxa=taxa, 
           titlefontfamily="helvetica oblique", titlefont=12,
           xticks=(1.5:5:11.5, 0:5:10))

savefig("docs/img/primates-crK8-pps.pdf")


# Branch rates
using Serialization
chn3 = deserialize("docs/data/primates-GO0002376-oib-max10-chn-br.jls")

n = length(postwalk(tree))
M = PhyloBDP(DLG(λ=randn(n), μ=randn(n), κ=log.(zeros(n))), rootprior, tree, bound)
function mfun3(M, x, n)
    λ = x[:λ][1]
    μ = x[:μ][1]
    l = [λ ; x.value[1,4:4+n-2]]
    m = [μ ; x.value[1,4+n-1:4+2n-3]]
    M(rates=DLG(λ=l, μ=m, κ=fill(-20., n)))
end
pp3= DeadBird.simulate(y->mfun3(M, y, n), data, chn3, 1000)

p2 = plot(pp3, size=(700,500), color=:gray, xscale=:identity, ylim=(-Inf, 0.5),
          xlim=(1,12), taxa=taxa, titlefontfamily="helvetica oblique",
          titlefont=12, xticks=(1.5:5:11.5, 0:5:10))

p1 = plot!(p2, pp, color=:salmon, size=(700,500), xscale=:identity, 
           ylim=(-Inf, 0.5), xlim=(1,12), taxa=taxa, 
           titlefontfamily="helvetica oblique", titlefont=12,
           xticks=(1.5:5:11.5, 0:5:10))


p0 = plot(pp3, size=(700,500), color=:gray, xscale=:identity, 
          ylim=(-2.5, 0.1), 
          xlim=(1,6), taxa=taxa, titlefontfamily="helvetica oblique",
          titlefont=12, xticks=(1.5:1:5.5, 0:1:4))

plot!(p0, pp, color=:salmon, size=(700,500), xscale=:identity, 
           taxa=taxa, ylim=(-2.5,0.1), ms=4,
           titlefontfamily="helvetica oblique", titlefont=12)

savefig("docs/img/primates-br-vs-cr-pps.pdf")


function getcolors(chn, tree, s, cs=ColorSchemes.viridis, transform=identity)
    xs, ys, d = NewickTree.treepositions(tree)
    colors = Matrix{typeof(get(cs,0.))}(undef, size(xs)...)
    rates = log.(zeros(size(xs)))
    for i=1:size(xs,2), j=[1,3]
        x = xs[j,i]
        ns = [k for (k,v) in d if v[1] == x]
        length(ns) == 0 && continue
        n = ns[1]-1
        v = "$s[$n]"
        r = mean(transform.(chn[v]))
        rates[j,i] = r
    end
    mn = minimum(filter(isfinite, rates))
    vs = similar(rates)
    for i in eachindex(rates)
        vs[i] = rates[i] - mn
        !isfinite(rates[i]) && (vs[i] = 0.)
    end
    mx = maximum(vs)
    vs ./= mx
    for i=1:size(xs, 2)
        vs[2,i] = (vs[1,i] + vs[3,i])/2
    end
    colors = [get(cs, vs[i]) for i in eachindex(vs)]
    colors = reshape(colors, size(vs))
    return colors, rates
end
cs, rs = getcolors(chn3, ftree, "l")

cs, ls = getcolors(chn3, ftree, "l")
p1 = plot(ftree, linecolor=cs, lw=2, fontfamily="helvetica oblique", pad=0.7,
          xlim=(-0.1,1.3))
cs, ms = getcolors(chn3, ftree, "m")
p2 = plot(ftree, linecolor=cs, lw=2, fontfamily="helvetica oblique", pad=0.7,
          xlim=(-0.1,1.3))
p3 = plot(p1, p2, size=(500,250))

for p in p0.subplots
    p.attr[:fontfamily_subplot] = "sans-serif"
end
plot(p0, plot(p1, p2, layout=(2,1)), layout=grid(1,2,widths=[0.7,0.3]),
     size=(900,400), bottom_margin=3mm, left_margin=3mm)

savefig("docs/img/primates-br-tree.pdf")

range(xs) = begin
    xs = filter(isfinite, xs)
    exp.([minimum(xs), mean(xs), maximum(xs)])
end
range(ls)
range(ms)

chn4 = deserialize("docs/data/primates-GO0002376-oib-max10-chn-crit.jls")

n = length(postwalk(tree))
M = PhyloBDP(DLG(λ=randn(n), μ=randn(n), κ=log.(zeros(n))), rootprior, tree, bound)
function mfun4(M, x, n)
    λ = x[:λ][1]
    l = [λ ; x.value[1,3:3+n-2]]
    M(rates=DLG(λ=l, μ=l, κ=fill(-20., n)))
end
pp4= DeadBird.simulate(y->mfun4(M, y, n), data, chn4, 1000)

p2 = plot(pp3, size=(700,500), color=:gray, xscale=:identity, ylim=(-Inf, 0.5),
          xlim=(1,12), taxa=taxa, titlefontfamily="helvetica oblique",
          titlefont=12, xticks=(1.5:5:11.5, 0:5:10))

p1 = plot!(p2, pp4, color=:salmon, size=(700,500), xscale=:identity, 
           ylim=(-Inf, 0.5), xlim=(1,12), taxa=taxa, 
           titlefontfamily="helvetica oblique", titlefont=12,
           xticks=(1.5:5:11.5, 0:5:10))

cs, ls = getcolors(chn4, ftree, "l")
p1 = plot(ftree, linecolor=cs, lw=2, fontfamily="helvetica oblique", pad=0.7,
          xlim=(-0.1,1.3))

