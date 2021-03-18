using CSV, DataFrames, DeadBird, NewickTree, Test
const datadir = joinpath(@__DIR__, "../example")
readtree = readnw ∘ readline

df = CSV.read(joinpath(datadir, "dicots/9dicots-f01-25.csv"), DataFrame)
tr = readtree(joinpath(datadir, "dicots/9dicots.nw"))

@testset "q = 0 tests" begin 
    d0, bound = CountDAG(df, tr)
    rates = DeadBird.ConstantDLGWGM(λ=0.1, μ=0.1, κ=0.1);
    m0 = PhyloBDP(rates, ShiftedGeometric(0.7), tr, bound);
    m1 = DeadBird.insertwgms(m0, 12=>(0.1, 2)); 
    m2 = DeadBird.insertwgms(m0, 12=>(0.1, 2), 6=>(0.3, 4), 6=>(0.35, 2)); 
    m3 = DeadBird.insertwgms(m0, 12=>(0.1, 2), 12=>(0.1,3), 6=>(0.3, 4), 6=>(0.35, 2)); 
    d1, _ = CountDAG(df, m1.order[end])
    d2, _ = CountDAG(df, m2.order[end])
    d3, _ = CountDAG(df, m3.order[end])
    l0 = DeadBird.loglikelihood(m0, d0)
    l1 = DeadBird.loglikelihood(m1, d1)
    l2 = DeadBird.loglikelihood(m2, d2)
    @test l0 ≈ l1 ≈ l2
end

# Yeast data set
# ==============
using Turing

df = CSV.read(joinpath(datadir, "ygob/ygob.N0.tsv"), DataFrame)
tr = readtree(joinpath(datadir, "ygob/ygob-12taxa.nw"))
for n in postwalk(tr); n.data.distance /= 100; end
x = DeadBird.getextra(df, tr)
counts = x.df[!,x.cols]

X0, bound = DeadBird.CountDAG(counts, tr)
r0 = DeadBird.ConstantDLG(λ=0.1, μ=0.1, κ=0.1);
m0 = PhyloBDP(r0, BetaGeometric(0.94, 4.), tr, bound, cond=:none);

n = getlca(tr, "Scerevisiae", "Vpolyspora")
r1 = DeadBird.ConstantDLGWGM(λ=0.1, μ=0.1, κ=0.1);
m1 = PhyloBDP(r1, BetaGeometric(0.94, 4.), tr, bound, cond=:none);
m1 = DeadBird.insertwgms(m1, id(n)=>(0.0229, 2));
X1, _ = CountDAG(counts, m1.order[end])

@info "" logpdf(m0, X0) logpdf(m1, X1)

@model nowgd(model, X) = begin
    η ~ Beta()
    μ ~ Turing.FlatPos(0.)
    λ = (1-η)*μ
    X ~ model(rates = ConstantDLG(λ=λ, μ=μ, κ=λ))
end

@model yeastwgd(model, X) = begin
    η ~ Beta()
    μ ~ Turing.FlatPos(0.)
    q ~ Beta()
    λ = (1 - η)*μ
    r = DeadBird.ConstantDLGWGM(λ=λ, μ=μ, κ=λ, q=Dict(0x0018 => q)); 
    X ~ model(rates=r)
end

@model yeastwgd_fq(model, X, q) = begin
    η ~ Beta()
    μ ~ Turing.FlatPos(0.)
    λ = (1 - η)*μ
    r = DeadBird.ConstantDLGWGM(λ=λ, μ=μ, κ=λ, q=Dict(0x0018 => typeof(η)(q))); 
    X ~ model(rates=r)
end

# The posterior seems to be bimodal at q == 0 vs. q == 1 !

# The LRT strongly favors the q=1 model
# 
# julia> optimize(yeastwgd_fq(m1, X1, 0.0001), MLE())
# ModeResult with maximized lp of -7937.90
# 2-element Named Vector{Float64}
# A  │
# ───┼─────────
# :η │ 0.931019
# :μ │  0.42166
# 
# julia> optimize(yeastwgd_fq(m1, X1, 1.), MLE())
# ModeResult with maximized lp of -7758.79
# 2-element Named Vector{Float64}
# A  │
# ───┼─────────
# :η │ 0.989152
# :μ │  1.74725
#
# The rate estimates with the WGD also closely align with the rates estimated
# for the yeasts that did not undergo WGD alone (cfr. two-type paper)

## Ferns

pt = "/home/arzwa/research/heche-ferns/"
df = CSV.read(joinpath(pt, "Orthogroups/Orthogroups.tsv"), DataFrame)
tr = readtree(joinpath(pt, "SpeciesTree_rooted.txt"))
for n in postwalk(tr); n.data.name = first(name(n), 3); end
rename!(df, first.(names(df), 3))
x  = DeadBird.getextra(df, tr)
y  = x.df[!,x.cols]
Xr = filter(x->all(Array(x) .< 100), y)
xs = StatsBase.counts(Matrix(Xr))

@model bgstat(X) = begin
    η ~ Beta()
    ζ ~ Turing.FlatPos(1.)
    X ~ BetaGeometric(η, ζ)
end

chn = sample(bgstat(xs), NUTS(), 100)
ζ, η = mean(chn).nt.mean

p = plot(grid=(:x), xticks=(1:100, vcat(1:5, ["..."])), ylim=(1/nrow(Xr),1), legend=:outertopright)
map(enumerate(eachcol(Xr))) do (k,yy)
    plot!(p, proportionmap(yy .+ 1), label=names(y)[k], xscale=:log10, yscale=:log10, alpha=0.6)
end; p
#plot!(proportionmap(vcat(Matrix(Xr)...) .+ 1), color=:black, linewidth=2)
plot!(1:100, x->pdf(BetaGeometric(η, ζ), x-1), color=:black, linewidth=2, linestyle=:dash)

Xr = Xr[sample(1:size(Xr, 1), 1000, replace=false),:]
X0, bound = DeadBird.CountDAG(Xr, tr)
r0 = DeadBird.ConstantDLG(λ=0.1, μ=0.1, κ=0.1);
m0 = PhyloBDP(r0, BetaGeometric(η, ζ), tr, bound, cond=:none);

n1 = getlca(tr, "Sal", "Azo")
n2 = getlca(tr, "Adi")
r1 = DeadBird.ConstantDLGWGM(λ=0.1, μ=0.1, κ=0.1);
m1 = PhyloBDP(r1, BetaGeometric(η, ζ), tr, bound, cond=:none);
m1 = DeadBird.insertwgms(m1, 
                         id(n1)=>(distance(n1)/2, 2), 
                         id(n2)=>(distance(n2)/2, 2));
X1, _ = CountDAG(Xr, m1.order[end])

@model model1(model, X) = begin
    η ~ Beta()
    α ~ Beta()
    μ ~ Turing.FlatPos(0.)
    q1 ~ Beta(0.1, 0.1)
    q2 ~ Beta(0.1, 0.1)
    ζ ~ Exponential()
    λ = (1 - α)*μ
    r = DeadBird.ConstantDLGWGM(λ=λ, μ=μ, κ=λ, 
                                q=Dict(0x000e => q1, 0x0010 => q2)); 
    p = BetaGeometric(η, 1+ζ)
    X ~ model(rates=r, rootp=p)
end

@model model2(model, X, q1, q2) = begin
    α ~ Beta()
    μ ~ Turing.FlatPos(0.)
    λ = (1 - α)*μ
    T = typeof(μ)
    r = DeadBird.ConstantDLGWGM(λ=λ, μ=μ, κ=λ, 
                                q=Dict(0x000e => T(q1), 0x0010 => T(q2))); 
    p = BetaGeometric(η, ζ)
    X ~ model(rates=r, rootp=p)
end


