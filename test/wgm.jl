using CSV, DataFrames, DeadBird, NewickTree, Test, Distributions
using DeadBird: ExcessConstantDLGWGM, ConstantDLGWGM, ConstantDLG 
using DeadBird: loglikelihood

const datadir = joinpath(@__DIR__, "example")
readtree = readnw ∘ readline

# works for default, not excess model...
@testset "q = 0 tests" begin 
    df = CSV.read(joinpath(datadir, "dicots/9dicots-f01-25.csv"), DataFrame)
    tr = readtree(joinpath(datadir, "dicots/9dicots.nw"))
    d0, bound = CountDAG(df, tr)
    rates = DeadBird.ConstantDLGWGM(λ=0.1, μ=0.1, κ=0.1);
    m0 = PhyloBDP(rates, ShiftedGeometric(0.7), tr, bound);
    m1 = DeadBird.insertwgms(m0, 12=>(0.1, 2, 0.)); 
    w2 = [12=>(0.1, 2, 0.), 6=>(0.3, 4, 0), 6=>(0.35, 2, 0.)]
    m2 = DeadBird.insertwgms(m0, w2...)
    m3 = DeadBird.insertwgms(m0, 12=>(0.1, 3, 0.), w2... )
    d1, _ = CountDAG(df, m1)
    d2, _ = CountDAG(df, m2)
    d3, _ = CountDAG(df, m3)
    l0 = DeadBird.loglikelihood(m0, d0)
    l1 = DeadBird.loglikelihood(m1, d1)
    l2 = DeadBird.loglikelihood(m2, d2)
    @test l0 ≈ l1 ≈ l2
end

@testset "Excess model with WGMs" begin
    df = CSV.read(joinpath(datadir, "ygob/ygob.N0.tsv"), DataFrame)
    tr = readtree(joinpath(datadir, "ygob/ygob-12taxa.nw"))
    x = DeadBird.getextra(df, tr)
    counts = x.df[1:1,x.cols]
    X0, bound = CountDAG(counts, tr)
    r0 = ConstantDLG(λ=0.1, μ=0.1, κ=0.1);
    m0 = PhyloBDP(r0, BetaGeometric(0.94, 4.), tr, bound, cond=:none);
    n = getlca(tr, "Scerevisiae", "Vpolyspora")
    r1 = ConstantDLGWGM(λ=0.1, μ=0.1, κ=0.1, excess=true);
    m1 = PhyloBDP(r1, BetaGeometric(0.94, 4.), tr, bound, cond=:none);
    m1 = DeadBird.insertwgms(m1, id(n)=>(0.0229, 2, 0.));
    X1, _ = CountDAG(counts, m1)
    l0 = loglikelihood(m0, X0)
    l1 = loglikelihood(m1, X1)
    @info "" l0 l1
    @test l0 ≈ l1
end

@testset "WGD excess model simulation and gradient" begin
    using ForwardDiff
    tr = readtree(joinpath(datadir, "ygob/ygob-12taxa.nw"))
    rootp = BetaGeometric(0.94, 4.)
    n = getlca(tr, "Scerevisiae", "Vpolyspora")
    l = length(postwalk(tr))
    r = DLGWGM(λ=zeros(l), μ=zeros(l), κ=zeros(l), excess=true)
    model = PhyloBDP(r, rootp, tr, 1, cond=:none)
    model = DeadBird.insertwgms(model, id(n)=>(0.0229, 2, 0.99));
    sdata = simulate(model, 100);
    dag, bound = CountDAG(sdata, model)
    model = model(bound=bound)                    
    function gradfun(x, model, data)
        r = ConstantDLGWGM(λ=x[1], μ=x[2], κ=x[1], q=Dict(0x0018 => x[3]), excess=true)
        m = model(rates=r)
        return loglikelihood(m, data)
    end
    y = [0.1, 1.7, 0.01]
    ∇ℓd = ForwardDiff.gradient(x->gradfun(x, model, dag), y)
    @test all(isfinite.(∇ℓd))
end


# Yeast data set
# ==============
using Turing, Plots, StatsPlots

@model nowgd_excess(model, X) = begin
    η ~ Beta()
    μ ~ Turing.FlatPos(0.)
    λ = (1-η)*μ
    X ~ model(rates = ConstantDLG(λ=λ, μ=μ, κ=λ))
end

@model yeastwgd_excess(model, X) = begin
    η ~ Beta()
    μ ~ Turing.FlatPos(0.)
    q ~ Beta()
    λ = (1 - η)*μ
    r = ExcessConstantDLGWGM(λ=λ, μ=μ, κ=λ, q=Dict(0x0018 => q)); 
    X ~ model(rates=r)
end

@model yeastwgd_total(model, X) = begin
    μ ~ Turing.FlatPos(0.)
    λ ~ Turing.FlatPos(0.)
    q ~ Beta()
    r = ConstantDLGWGM(λ=λ, μ=μ, κ=zero(λ + 1e-10), q=Dict(0x0018 => q)); 
    X ~ model(rates=r)
end

@model yeastwgd_fq(model, X, q) = begin
    η ~ Beta()
    μ ~ Turing.FlatPos(0.)
    λ = (1 - η)*μ
    r = ExcessConstantDLGWGM(λ=λ, μ=μ, κ=λ, q=Dict(0x0018 => typeof(η)(q))); 
    X ~ model(rates=r)
end

function mfun0(chn)
    function fun(x)
        θ = get(chn[1], [:η, :μ])
        λ = (1. - θ.η[1]) * θ.μ[1]
        PhyloBDP(ConstantDLG(λ=λ, μ=θ.μ[1], κ=λ), 
                 BetaGeometric(0.94, 4.), tr, bound, cond=:none)
    end
end

function mfun1(chn, model)
    function fun(x)
        θ = get(chn[1], [:q, :η, :μ])
        λ = (1. - θ.η[1]) * θ.μ[1]
        q = Dict(0x0018 => θ.q[1])
        rates = ExcessConstantDLGWGM(λ=λ, μ=θ.μ[1], κ=λ, q=q)
        model(rates=rates)
    end
end

# Simulations
# -----------
tr = readtree(joinpath(datadir, "ygob/ygob-12taxa.nw"))

# no WGD as reference
rootp = BetaGeometric(0.94, 4.)
model = PhyloBDP(ConstantDLG(λ=0.1, μ=1.7, κ=0.1), rootp, tr, 1, cond=:none)
sdata = simulate(model, 1000);
dag, bound = CountDAG(sdata, model)
model = model(bound=bound)
chain = sample(nowgd_excess(model, dag), NUTS(), 1000) 
sims = simulate(mfun0(chain), sdata, chain)
plot(sims)

# WGD total
rootp = ShiftedBetaGeometric(0.94, 4.)
n = getlca(tr, "Scerevisiae", "Vpolyspora")
model = PhyloBDP(ConstantDLGWGM(λ=0.2, μ=0.2, κ=0.0), rootp, tr, 2)
model = DeadBird.insertwgms(model, id(n)=>(0.0229, 2, 0.2));
sdata = simulate(model, 200);
dag, bound = CountDAG(sdata, model)
model = model(bound=bound)                    
chain = sample(yeastwgd_total(model, dag), NUTS(), 500) 

# with WGD, excess
rootp = BetaGeometric(0.94, 4.)
n = getlca(tr, "Scerevisiae", "Vpolyspora")
model = PhyloBDP(ConstantDLGWGM(λ=0.1, μ=1.5, κ=0.1, excess=true), rootp, tr, 1, cond=:none)
model = DeadBird.insertwgms(model, id(n)=>(0.0229, 2, 1.));
sdata = simulate(model, 200);
dag, bound = CountDAG(sdata, model)
model = model(bound=bound)                    
chain = sample(yeastwgd_excess(model, dag), NUTS(), 1000) 
sims = simulate(mfun1(chain, model), sdata, chain)

# The actual data
# ---------------
df = CSV.read(joinpath(datadir, "ygob/ygob.N0.tsv"), DataFrame)
x = DeadBird.getextra(df, tr)
counts = x.df[:,x.cols]
X0, bound = CountDAG(counts, tr)
n = getlca(tr, "Scerevisiae", "Vpolyspora")
r1 = ExcessConstantDLGWGM(λ=0.1, μ=0.1, κ=0.1);
m1 = PhyloBDP(r1, BetaGeometric(0.94, 4.), tr, bound, cond=:none);
m1 = DeadBird.insertwgms(m1, id(n)=>(0.0229, 2, 0.2));
X1, _ = CountDAG(counts, m1)

c1 = sample(yeastwgd_excess(m1, X1), NUTS(), 1000) 

sims = simulate(mfun1(c1, m1), counts, c1)

using Optim
ml1 = optimize(yeastwgd_fq(m1, X1, 0.0001), MLE()) 
ml2 = optimize(yeastwgd_fq(m1, X1, 0.9999), MLE()) 
ml3 = optimize(yeastwgd(m1, X1), MLE()) 

@model yeastwgdp(model, X) = begin
    η ~ Beta()
    ζ ~ Exponential(4.)
    μ ~ Turing.FlatPos(0.)
    q ~ Beta()
    λ = (1 - η)*μ
    p = BetaGeometric(η, ζ+1)
    r = ExcessConstantDLGWGM(λ=λ, μ=μ, κ=λ, q=Dict(0x0018 => q)); 
    X ~ model(rates=r, rootp=p)
end

function mfun2(chn, model)
    function fun(x)
        θ = get(chn[1], [:q, :ζ, :η, :μ])
        λ = (1. - θ.η[1]) * θ.μ[1]
        q = Dict(0x0018 => θ.q[1])
        r = ExcessConstantDLGWGM(λ=λ, μ=θ.μ[1], κ=λ, q=q)
        p = BetaGeometric(θ.η[1], θ.ζ[1]+1)
        model(rates=r, rootp=p)
    end
end

chain2 = sample(yeastwgdp(m1, X1), NUTS(), 1000)
sims2 = simulate(mfun2(chain2, m1), counts, chain2)

@model yeastwgd_fqp(model, X, q) = begin
    η ~ Beta()
    ζ ~ Exponential(4.)
    μ ~ Turing.FlatPos(0.)
    λ = (1 - η)*μ
    p = BetaGeometric(η, ζ+1)
    r = ExcessConstantDLGWGM(λ=λ, μ=μ, κ=λ, q=Dict(0x0018 => typeof(η)(q))); 
    X ~ model(rates=r, rootp=p)
end

chain3 = sample(yeastwgd_fqp(m1, X1, 1.0), NUTS(), 1000)
sims2 = simulate(mfun2(chain2, m1), counts, chain2)

## Ferns
using StatsBase
pt = "/home/arzwa/research/heche-ferns/"
df = CSV.read(joinpath(pt, "Orthogroups/Orthogroups.tsv"), DataFrame)
tr = readtree(joinpath(pt, "SpeciesTree_rooted.txt"))
for n in postwalk(tr); n.data.name = first(name(n), 3); end
rename!(df, first.(names(df), 3))
x  = DeadBird.getextra(df, tr)
y  = x.df[!,x.cols]
Xr = filter(x->all(Array(x) .< 20), y)
xs = StatsBase.counts(Matrix(Xr))

@model bgstat(X) = begin
    η ~ Beta()
    ζ ~ Turing.FlatPos(1.)
    X ~ BetaGeometric(η, ζ)
end

chn = sample(bgstat(xs), NUTS(), 100)
ζ, η = mean(chn).nt.mean

p = plot(grid=(:x), xticks=(1:100, vcat(1:5, ["..."])), 
         ylim=(1/nrow(Xr),1), legend=:outertopright)
map(enumerate(eachcol(Xr))) do (k,yy)
    plot!(p, proportionmap(yy .+ 1), label=names(y)[k], 
          xscale=:log10, yscale=:log10, alpha=0.6)
end; p
#plot!(proportionmap(vcat(Matrix(Xr)...) .+ 1), color=:black, linewidth=2)
plot!(1:100, x->pdf(BetaGeometric(η, ζ), x-1), 
      color=:black, linewidth=2, linestyle=:dash)

Xr = Xr[sample(1:size(Xr, 1), 1000, replace=false),:]
X0, bound = CountDAG(Xr, tr)
r0 = ConstantDLG(λ=0.1, μ=0.1, κ=0.1);
m0 = PhyloBDP(r0, BetaGeometric(η, ζ), tr, bound, cond=:none);

n1 = getlca(tr, "Sal", "Azo")
n2 = getlca(tr, "Adi")
r1 = ExcessConstantDLGWGM(λ=0.1, μ=0.9, κ=0.1);
m1 = PhyloBDP(r1, BetaGeometric(η, ζ), tr, bound, cond=:none);
m1 = DeadBird.insertwgms(m1, 
                         id(n1)=>(distance(n1)/2, 2, 0.1), 
                         id(n2)=>(distance(n2)/2, 2, 0.1));
X1, _ = CountDAG(Xr, m1)

@model model1(model, X) = begin
    α ~ Beta()
    μ ~ Turing.FlatPos(0.)
    q1 ~ Beta()
    q2 ~ Beta()
    λ = (1 - α)*μ
    r = ExcessConstantDLGWGM(λ=λ, μ=μ, κ=λ, 
                             q=Dict(0x000e => q1, 0x0010 => q2)); 
    X ~ model(rates=r)
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


