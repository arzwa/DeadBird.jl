using Pkg; Pkg.activate(joinpath(@__DIR__, "docs"))
using DeadBird
using Distributions, Turing, CSV, DataFrames, NewickTree, Optim, Parameters
using Plots, StatsPlots, Measures, StatsBase

dir = "docs/data/dicots"
tree = readnw(readline(joinpath(dir, "9dicots.nw")))
node = postwalk(tree)[4]
λ = μ = 1.5
η = 0.75
p = ShiftedGeometric(η)

# Models
@model constantrate0(M, dag, ::Type{T}=Float64) where T = begin
    λ ~ Exponential(1.5)
    μ ~ Exponential(1.5)
    dag ~ M(rates=ConstantDLG(λ=λ, μ=μ, κ=T(1e-10)))
end

@model constantrate1(M, dag, ::Type{T}=Float64) where T = begin
    λ ~ Exponential(1.5)
    μ ~ Exponential(1.5)
    q ~ Beta() 
    dag ~ M(rates=ConstantDLGWGM(λ=λ, μ=μ, κ=T(1e-10), q=Dict(0x0012=>q)))
end

# Example ignoring heterogeneity
n = length(postwalk(tree))
l = rand(Normal(log(λ), 0.5), n)
m = l .+ rand(Normal(0, 0.5), n)
l[5] = quantile(l, 0.75)
m[5] = quantile(m, 0.25)

l = [0.5, -0.04, 0.68, 0.3, 0.68, -0.28, -0.39, 0.26, -0.28, 0.56, 0.91, 0.21, 0.88, 0.16, 0.92, 0.17, 0.31]
m = [-0.7, 0.93, 1.15, 0.15, -0.21, -0.88, -0.91, -0.07, 0.28, 1.11, 1.0, 0.45, 0.25, -0.25, 1.41, -0.21, 0.88]

#col = fill(:black, n)
#col[5] = :salmon
#p0 = scatter(exp.(l), exp.(m), color=col, grid=false, framestyle=:box,
#             legend=false, size=(250,250), xlabel="\$\\lambda\$", ylabel="\$\\mu\$")

using Random
Random.seed!(123)
M = PhyloBDP(DLG(λ=l, μ=m, κ=fill(-Inf, n)), p, tree, 1)
df = DeadBird.simulate(M, 1000)
df = df[:,1:9]

model1 = PhyloBDP(ConstantDLGWGM(λ=λ, μ=μ, κ=0.), p, tree, 10)
model1 = DeadBird.insertwgms(model1, id(node)=>(distance(node)/2, 2, 0.15))
dag, bound = CountDAG(df, model1)
model1 = model1(bound=bound)
res1 = optimize(constantrate1(model1, dag), MLE())

model0 = PhyloBDP(ConstantDLG(λ=λ, μ=μ, κ=0.), p, tree, bound)
dag, bound = CountDAG(df, model0)
res0 = optimize(constantrate0(model0, dag), MLE())

Λ = 2*(res1.lp - res0.lp)

default(title_loc=:left, titlefont=9, guidefont=9)
p0 = scatter(exp.(l), exp.(m), color=col, grid=false, framestyle=:box, title="(C)",
             legend=false, xlabel="\$\\lambda\$", ylabel="\$\\mu\$", left_margin=0mm)
p1 = plot(tree, pad=0., xlim=(0,0.16), right_margin=0mm, title="(A)")
scatter!(p1, [0.03], [2.5])
XX = Matrix(df[:,name.(getleaves(tree))]) |> permutedims
o = reverse(sortperm(vec(sum(XX, dims=1))))
XX = XX[:,o]
p2 = heatmap(XX, color=:binary, framestyle=:box, yticks=false, grid=false, title="(B)",
             left_margin=0mm, xlabel="families")
plot(p1, p2, p0, layout=grid(1,3,widths=[0.15,0.58,0.27]), size=(600,200), bottom_margin=5mm)


# branch rates
@model branchrates1(M, n, dag, ::Type{T}=Float64) where T = begin
    r ~ Normal(log(1.5), 1.)
    τ ~ Exponential()
    λ ~ MvNormal(fill(r, n-1), τ)
    μ ~ MvNormal(fill(r, n-1), τ)
    κ = fill(T(-10.), n)
    q ~ Beta() 
    dag ~ M(rates=DLGWGM(λ=[r; λ], μ=[r; μ], κ=κ, q=Dict(0x0012=>q)))
end

n = length(postwalk(tree))
model1 = PhyloBDP(DLGWGM(λ=l, μ=m, κ=fill(-Inf, n)), p, tree, 10)
model1 = DeadBird.insertwgms(model1, id(node)=>(distance(node)/2, 2, 0.15))
dag, bound = CountDAG(df, model1)
model1 = model1(bound=bound)

chn1 = sample(branchrates1(model1, n, dag), NUTS(), 200; save_state=true)
chn1 = sample(branchrates1(model1, n, dag), NUTS(), 400; save_state=true, resume_from=chn0)

chn1 = sample(branchrates1(model1, n, dag), NUTS(), 20; save_state=true)


postdf = DataFrame(chn1)

function reflected_kde(xs; kwargs...)
    K = kde([xs ; -xs]; kwargs...)
    n = length(K.x) ÷ 2
    K.density = K.density[n+1:end] .* 2
    K.x = K.x[n+1:end]
    return K
end


default(title_loc=:left, titlefont=9, guidefont=9, grid=false, framestyle=:box, legend=false)
ps = map(["λ", "μ"]) do s
    p1 = plot()
    map(1:n-1) do i
        po = postdf[:,"$s[$i]"]
        x = exp(mean(po))
        e1 = x - exp(quantile(po, 0.025))
        e2 = exp(quantile(po, 0.975)) - x
        y = s == "λ" ? l : m
        lab1 = "\$\\$(s=="λ" ? "hat{\\lambda}" : "hat{\\mu}")\$"
        lab2 = "\$\\$(s=="λ" ? "lambda" : "mu")\$"
        scatter!(p1, [exp(y[i+1])], [x], yerr=([e1], [e2]), color=:black, ms=3,
                 xlim=(0,6), ylim=(0,6), title=lab1, xlabel=lab2)
    end
    plot!(p1, x->x, color=:gray, alpha=0.5, lw=2)
    p1
end 
p3 = histogram(postdf[:,:q], color=:white, norm=:true, xlabel="\$q\$",
               xlim=(0,0.2), bins=-0.01:0.01:0.2, ylabel="\$p\$")
plot!(reflected_kde(postdf[:,:q], bandwidth=0.01), color=:salmon, lw=2)
p4 = plot(postdf[:,:q], color=:black, xlabel="iteration", ylabel="\$q\$")
p5 = plot(p3, p4, layout=(2,1))
plot(ps..., p5, size=(700,250), layout=(1,3), bottom_margin=3mm)

