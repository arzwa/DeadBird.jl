using Pkg; Pkg.activate(joinpath(@__DIR__, "../"))
using BirdDad, NewickTree, CSV, Turing, DataFrames, StatsBase
import BirdDad: CountDAG, ConstantDLG, PhyloBDP, mle_problem, RatesModel, DLG

# ## The constant-rates model

# Read in the data and tree
df = CSV.read(joinpath(@__DIR__, "../../example/9dicots-f01-100.csv"))
tree = readnw(readline(joinpath(@__DIR__, "../../example/9dicots.nw")))

# Construct the data object
dag, bound = CountDAG(df, tree)

# Construct the model
rates = RatesModel(ConstantDLG(λ=.1, μ=.1, κ=0.0, η=1/1.5), fixed=(:κ,))
model = PhyloBDP(rates, tree, bound)

@model constantrates(dag, model) = begin
    r ~ MvLogNormal(ones(2))
    η ~ Beta(2,1)
    dag ~ model((λ=r[1], μ=r[2], η=η))
end

bmodel = constantrates(dag, model)
chain = sample(bmodel, NUTS(0.65), 1000);
posterior = DataFrame(chain)

# # Posterior predictive checks - some exploration
using FakeFamily, Parameters
import FakeFamily: dlsimbunch

struct _PostPredSim{T}
    model    ::T
    posterior::DataFrame
    paramfun ::Function
end

function sim(pp::_PostPredSim, N, M;
        sumfun=identity, profiles=true, trees=false)
    @unpack posterior, model, paramfun = pp
    n, m = size(pp.posterior)
    ts, ps = dlsimbunch(model, 1, minn=2)
    Y = typeof(sumfun(ps))[]
    X = typeof(ps)[]
    T = typeof(ts)[]
    for i=1:N
        x = posterior[rand(1:n),:]
        model′ = paramfun(model, x)
        ts, ps = dlsimbunch(model′, M, minn=2)
        sumfun != identity && push!(Y, sumfun(ps))
        profiles && push!(X, ps)
        trees && push!(T, ts)
    end
    (stats=Y, profiles=X, trees=T)
end

function leafpmf(df, xmax)
    function f(column)
        xs = counts(column, 0:xmax)
        exceeding = length(filter(x->x>xmax, column))
        [xs ; exceeding]
    end
    Dict(col=>f(df[!,col]) for col in names(df))
end

function pppval(y, ys)
    p = sum(y .> ys)/length(ys)
    p > 0.5 ? one(p) - p : p
end

paramfun(model, x) = model((λ=x[1], μ=x[2], η=x[3]))
pp = _PostPredSim(model, posterior, paramfun)
Y, _, _ = sim(pp, 1000, nrow(df), sumfun=x->leafpmf(x, 3))
Yobs = leafpmf(df, 3)

function leafpmfplot(Y, Yobs; sze=(500,700), kwargs...)
    ps = []
    for (k, sp) in enumerate(keys(Yobs))
        data = hcat([y[sp] for y in Y]...)
        pss = [
            begin
                ys = data[i,:]
                y  = Yobs[sp][i]
                p = stephist(ys, color=:black, showaxis=false,
                    nbins=20, fill=true, alpha=0.1)
                vline!([y],
                    color=pppval(y, ys) < 0.05 ? :firebrick : :gray,
                    linewidth=2, xticks=false, yticks=false)
            end for i=1:size(data)[1]]
        ylabel!(pss[1], sp, title_loc=:left, titlefont=9)
        if k == 1
            for i=0:length(pss)-1
                title!(pss[i+1], "$i genes",
                    title_loc=:left, titlefont=8)
            end
            title!(pss[end], "≧ $(length(pss)-1) genes",
                title_loc=:left, titlefont=8)
        end
        p = plot(pss...,
            layout=(1, size(data)[1]),
            legend=false, grid=false)
        push!(ps, p)
    end
    ncol = length(collect(values(Yobs))[1])
    plot(ps..., layout=(length(ps),1), size=sze, kwargs...)
end


# # Critical branch rates
n = length(postwalk(tree))
rates = RatesModel(DLG(λ=zeros(n), μ=zeros(n), κ=0.0, η=1/1.5), fixed=(:κ,))
model = PhyloBDP(rates, tree, bound)
@model criticalrates(dag, model, ::Type{T}=Float64) where T = begin
    τ ~ truncated(Cauchy(0, 1), 0, Inf)
    r = Vector{T}(undef, n)
    r[1] ~ Normal(-1., 2)
    for i=2:n
        r[i] ~ Normal(r[1], τ)
    end
    η ~ Beta(2,1)
    dag ~ model((λ=r, μ=r, η=η))
end

bmodel = criticalrates(dag, model)
chain = sample(bmodel, NUTS(0.65), 1000);
pdf2 = DataFrame(chain)

paramfun(model, x) = Array(x) |> x->model((λ=x[1:n], μ=x[1:n], η=x[end-1]))
pp = _PostPredSim(model, pdf2, paramfun)
Y, _, _ = sim(pp, 1000, nrow(df), sumfun=x->leafpmf(x, 4))
Yobs = leafpmf(df, 4)
leafpmfplot(Y, Yobs, sze=(800,800))

# # Bivariate rates model
n = length(postwalk(tree))
rates = RatesModel(DLG(λ=zeros(n), μ=zeros(n), κ=0.0, η=1/1.5), fixed=(:κ,))
model = PhyloBDP(rates, tree, bound)
@model bivariaterates(dag, model, ::Type{T}=Float64) where T = begin
    τ ~ truncated(Cauchy(0, 1), 0, Inf)
    ρ ~ Uniform(-1,1)
    R = [1. ρ ; ρ 1.]
    S = [τ 0. ; 0. τ]
    Σ = S*R*S
    r = Matrix{T}(undef, 2, n)
    r[:,1] ~ MvNormal(-ones(2), 2)
    for i=2:n
        r[:,i] ~ MvNormal(r[:,1], Σ)
    end
    η ~ Beta(2,1)
    dag ~ model((λ=r[1,:], μ=r[2,:], η=η))
end

bmodel = bivariaterates(dag, model)
chain = sample(bmodel, NUTS(0.65), 1000);
pdf3 = DataFrame(chain)

paramfun(model, x) = Array(x) |> x->model((λ=x[1:n], μ=x[n+1:2n], η=x[end-2]))
pp = _PostPredSim(model, pdf3, paramfun)
Y, _, _ = sim(pp, 1000, nrow(df), sumfun=x->leafpmf(x, 4))
Yobs = leafpmf(df, 4)
leafpmfplot(Y, Yobs, sze=(800,800))

# it does not seem very relevant to look for better test quantities, since the pmf's for the number of genes at te leaves of the tree seems to serve as a good visual diagnostic.
