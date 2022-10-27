using Pkg; Pkg.activate(joinpath(@__DIR__, "docs"))
using DeadBird
using Distributions, Turing, CSV, DataFrames, NewickTree, Optim, Parameters
using Plots, StatsPlots, Measures, StatsBase

dir = "docs/data/dicots"
data = CSV.read("$dir/9dicots-f01-1000.csv", DataFrame)
tree = readnw(readline(joinpath(dir, "9dicots.nw")))
taxa = Dict("vvi"=>"V. vinifera", 
            "ath"=>"A. thaliana", 
            "cpa"=>"C. papaya", 
            "mtr"=>"M. truncatula", 
            "ptr"=>"P. trichocarpa", 
            "bvu"=>"B. vulgaris", 
            "cqu"=>"C. quinoa", 
            "ugi"=>"U. gibba", 
            "sly"=>"S. lycopersicum")

mx = maximum(Matrix(data))-1

# get a named tuple for a chain 'row'
function getparams(x)
    vars = x.value.axes[2]
    (; [var=>x for (var,x) in zip(vars, vec(x.value.data))]...)
end

function makevec(tup, s)
    ks = string.(keys(tup))
    [tup[i] for i in filter(i->startswith(ks[i], s), 1:length(tup))]
end


# We will use the DAG data structure (most efficient, but admits no
# family-specific models).
dag, bound = CountDAG(data, tree)

# based on BG fit
const ETA = 0.92
const ZETA = 10.

# 0. BG fit
# =========
# estimate ζ
@model bgfit(xs) = begin
    η ~ Beta()
    ζ ~ Exponential(10.)
    M = ShiftedBetaGeometric(η, ζ+1)
    for (k,x) in enumerate(xs)
        Turing.@addlogprob!(x*logpdf(M,k))
    end
end
function bgsim(chain, Y)
    modfun(x) = ShiftedBetaGeometric(get(x, :η).η[1], get(x, :ζ).ζ[1])
    Ys = map(i->rand(modfun(chain[i]), sum(Y)), 1:length(chain))
    Ys = mapreduce(x->proportions(x, 1:length(Y)), hcat, Ys)
    DeadBird.quantiles(Ys)
end

xs = map(sp->counts(filter(x-> x > zero(x), data[:,sp])), [:cpa, :vvi, :mtr])
X = xs[3]
chn = sample(bgfit(X), NUTS(), 500)
pps = bgsim(chn, X)
y = log10.(X ./sum(X))
p = scatter(1.5:length(y) + 0.5, y, xscale=:log10, color=:black)
stepplot!(pps[:,1], ribbon=(pps[:,2], pps[:,3]), fillalpha=0.2)

# 1. Constant rates
# =================
@model constantrate(dag, bound, tree, ::Type{T}=Float64) where T = begin
    η ~ Beta()
    #ζ ~ LogNormal(0, 2.)
    ζ = 10.
    λ ~ Turing.FlatPos(0.)
    μ ~ Turing.FlatPos(0.)
    θ = ConstantDLG(λ=λ, μ=μ, κ=T(1e-10))
    p = ShiftedBetaGeometric(η, ζ)
    dag ~ PhyloBDP(θ, p, tree, bound)
end

chn1 = sample(constantrate(dag, bound, tree), NUTS(), 500)

function mfun1(x, tree, bound)
    @unpack λ, μ, η = getparams(x)
    ζ = 10. -1
    PhyloBDP(ConstantDLG(λ=λ, μ=μ, κ=0.), ShiftedBetaGeometric(η, ζ+1), tree, bound)
end
pp1 = DeadBird.simulate(y->mfun1(y, tree, bound), data, chn1, 1000)

p1 = plot(pp1, taxa=taxa, xscale=:identity, xlim=(1,mx), xticks=(1:2:mx, 0:2:mx))

serialize("docs/data/dicots/crate1.jls", chn1)
chn1 = deserialize("docs/data/dicots/crate1.jls")


# 2. Branch rates
# ===============
@model brate(dag, bound, tree, n=length(postwalk(tree)), ::Type{T}=Float64) where T = begin
    #η ~ Beta()
    #ζ ~ LogNormal(0, 1.)
    η = 0.92
    ζ = 10.
    τ = .1
    #τ ~ Exponential()
    r ~ Normal(log(3.), 2)
    λ ~ MvNormal(fill(r, n-1), τ)  
    μ ~ MvNormal(fill(r, n-1), τ)  
    l = [r ; λ]
    m = [r ; μ]
    θ = DLG(λ=l, μ=m, κ=fill(T(-10), n))
    p = ShiftedBetaGeometric(η, ζ)
    dag ~ PhyloBDP(θ, p, tree, bound)
end

chn2 = sample(brate(dag, bound, tree), NUTS(), 500, save_state=true)

serialize("docs/data/dicots/brate2.jls", chn2)

# currently we use this one
chn2 = deserialize("docs/data/dicots/brate3-tau0.1.jls")

function mfun2(x, tree, bound)
    xs = getparams(x)
    ζ = 10.
    η = 0.92
    l = [xs.r1 ; makevec(xs, "λ")]
    m = [xs.r2 ; makevec(xs, "μ")]
    PhyloBDP(DLG(λ=l, μ=m, κ=fill(-Inf, length(l))), ShiftedBetaGeometric(η, ζ), tree, bound)
end
pp2 = DeadBird.simulate(y->mfun2(y, tree, bound), data, chn2, 1000)


# Plot
# ====
# overall plot
default(guidefont=10, titlefont=10, title_loc=:left, framestyle=:default)
order = reverse(name.(getleaves(tree)))
p1 = plot(pp1, order=order, taxa=taxa, xscale=:identity, 
          xlim=(1,mx), xticks=(1.5:2:mx+0.5, 0:2:mx))
p2 = plot!(p1, order=order, pp2, taxa=taxa, color=:salmon, fillalpha=0.4, xscale=:identity,
           xlim=(1,mx), xticks=(1.5:2:mx+0.5, 0:2:mx))

# detail
ps = map(["cqu", "ptr"]) do sp
    rnge = 1:5
    cqu = permutedims(pp2.sims[sp][rnge,:])
    violin(cqu, color=:lightgray, linecolor=:lightgray, xticks=(rnge, rnge .-1))
    y = proportions(data[:,sp])[1:rnge[end]]
    sticks!(rnge, y, color=:black, ms=4)
    scatter!(rnge .+1e-2, y, color=:black, ms=5, title=taxa[sp])
end 
ylabel!.(ps, "\$p_n\$")
xlabel!(ps[2], "\$n\$")
p3 = plot(ps..., ylim=(0,0.7), legend=false, ygrid=false, size=(300,200),
          titlefont=10, titlefontfamily="helvetica oblique", title_loc=:left,
          layout=(2,1))

plot(p2, p3, size=(700,400), layout=grid(1,2,widths=[0.85,0.15]))


# colored tree
ftree = NewickTree.relabel(tree, taxa)

using ColorSchemes
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
    return colors, filter(isfinite, rates)
end

function plot_cbar!(p, z, cs=ColorSchemes.viridis; n=100, lw=10)
    x1, x2 = xlims(p)
    y1, y2 = ylims(p)
    d = (y2 - y1)/n
    plot!(p, fill(x2,n+1), y1:d:y2, color=get.(Ref(cs), 0:(1/n):1), lw=lw)
    annotate!(x2, y2+d, text(@sprintf("%.2f  ", z[2]), :right, 8))
    annotate!(x2, y1, text(@sprintf("%.2f  ", z[1]), :right, 8))
    xlims!(p, x1, x2+(x2-x1)*0.05) 
end

cs, ls = getcolors(chn2, ftree, "λ")
t1 = plot(ftree, linecolor=cs, lw=2, fontfamily="helvetica oblique", pad=0.7,
          xlim=(-0.0,0.3), title="\$\\lambda\$")
plot_cbar!(t1, exp.(extrema(ls)))

cs, ms = getcolors(chn2, ftree, "μ")
t2 = plot(ftree, linecolor=cs, lw=2, fontfamily="helvetica oblique", pad=0.7,
          xlim=(-0.0,0.3), title="\$\\mu\$")
plot_cbar!(t2, exp.(extrema(ms)))
tp = plot(t1, t2, size=(500,250), colobar=true)

tp = plot(t1, t2, layout=(2,1))
for sp in p2.subplots[1:6]
    xlabel!(sp, "")
end
for sp in [p2.subplots[7:end] ; p3.subplots[2]; tp.subplots[2]]
    sp.attr[:bottom_margin]=3mm
end
for sp in p2.subplots[[2,3,5,6,8,9]]
    ylabel!(sp, "")
end
for sp in p2.subplots[[1,4,7]]
    sp.attr[:left_margin]=3mm
end
pp = plot(p2, p3, tp, size=(1000,400), layout=grid(1,3,widths=[0.65,0.12,0.23]))
for sp in pp.subplots
    sp.attr[:fontfamily_subplot] = "sans-serif"
end
plot(pp)

savefig("docs/img/dicot-9taxa-ppd.pdf")


# Table
# =====

function getrates(chain, s)
    map(1:length(chain)) do i 
        makevec(getparams(chain[i]), s)
    end |> x->hcat(x...)
end

l = exp.(getrates(chn2, "λ"))
m = exp.(getrates(chn2, "μ"))
s  = "| Species | \$\\lambda\$ | 95% UI | \$\\mu\$ | 95% UI |\n"
s *= "| ------- | -----------: | -----: | -------: | -----: |\n"
for n in getleaves(ftree)
    i = id(n) - 1
    m1 = mean(l[i,:])
    q11, q12 = quantile(l[i,:], [0.025, 0.975])
    m2 = mean(m[i,:])
    q21, q22 = quantile(m[i,:], [0.025, 0.975])
    s *= @sprintf "| *%s* | %.1f | (%.1f, %.1f) | %.1f | (%.1f, %.1f) | \n" name(n) m1 q11 q12 m2 q21 q22
end
println(s)



# 3. Branch rates, with WGDs
# ==========================
@model bratewgd(model, n, dag, wgds, ::Type{T}=Float64) where T = begin
    τ = 0.1
    #τ ~ Exponential(0.2)
    r1 ~ Normal(log(1.5), 2)
    r2 ~ Normal(log(1.5), 2)
    λ ~ MvNormal(fill(r1, n-1), τ)  
    μ ~ MvNormal(fill(r2, n-1), τ)  
    l = [r1 ; λ]
    m = [r2 ; μ]
    q ~ filldist(Beta(), length(wgds))
    qdict = Dict(wgds[i]=>q[i] for i=1:length(wgds))
    θ = DLGWGM(λ=l, μ=m, κ=fill(T(-10.), n), q=qdict)
    dag ~ model(rates=θ)
end

ns = [getlca(tree, "ptr"),
      getlca(tree, "cqu"),
      getlca(tree, "mtr"),
      getlca(tree, "ath"),
      getlca(tree, "sly"),
      getlca(tree, "ugi"),
      getlca(tree, "cpa"),
      getlca(tree, "bvu"),
      getlca(tree, "vvi"),
     ]
wgds = [id(n)=>(distance(n)/2, 2, 0.1) for n in ns]

dag, bound = CountDAG(data, tree)

rootprior = ShiftedBetaGeometric(ETA, ZETA)
n = length(postwalk(tree))
model1 = PhyloBDP(DLGWGM(λ=zeros(n), μ=zeros(n), κ=fill(-Inf, n)), rootprior, tree, bound)
model1 = DeadBird.insertwgms(model1, wgds...)
dag, bound = CountDAG(data, model1)
model1 = model1(bound=bound)
wgds = sort(collect(keys(model1.rates.q)))

#chn0  = sample(bratewgd(model1, n, dag, wgds), NUTS(), 100; save_state=true)
#chn01 = sample(bratewgd(model1, n, dag, wgds), NUTS(), 500; save_state=true, resume_from=chn0)

#serialize("docs/data/dicots/brate-wgd2.jls", chn0)

chn3 = deserialize("docs/data/dicots/brate-wgd2.jls")

function mfun3(model, wgds, x)
    xs = getparams(x)
    l = [xs.r1 ; makevec(xs, "λ")]
    m = [xs.r2 ; makevec(xs, "μ")]
    q = makevec(xs, "q")
    qdict = Dict(wgds[i]=>q[i] for i=1:length(wgds))
    θ = DLGWGM(λ=l, μ=m, κ=fill(-Inf, length(l)), q=qdict)
    model(rates=θ)
end
pp3 = DeadBird.simulate(y->mfun3(model1, wgds, y), data, chn3, 1000)


# Ancestral state stuff
# =====================
mat, _ = ProfileMatrix(data, getroot(model1))
xs = DeadBird.sample_ancestral(y->mfun3(model1, wgds, y), chn3, mat, 1000)

xs = deserialize("docs/data/dicots/brate-wgd2-anc.jls")

# posterior expected number of genes at each node for each family
xs1 = permutedims(reshape(sum(xs, dims=3) ./ 1000, length(model1), size(mat,1)))

# average number of genes per node across families for each simulation
# replicate
xs2 = permutedims(reshape(sum(xs, dims=2) ./ 1000, length(model1), 1000))

xs3 = vec(sum(xs1, dims=1))

ps = NewickTree.treepositions(model1[1])
taxa2 = Dict(k=>@sprintf("         %s", v) for (k,v) in taxa)
p = plot(NewickTree.relabel(tree, taxa2), fontfamily="helvetica oblique")
for n in model1.order
    q1, q2 = quantile(xs2[:,id(n)], (0.025, 0.975)) .* 1000
    #E = sum(xs2[:,id(n)])
    aln = DeadBird.iswgmafter(n) || isleaf(n) ? :left : :right
    if isleaf(n)
        annotate!(p, ps[n][1], ps[n][2], text(@sprintf("%4d", q1) , :helvetica, 8, aln)) 
    else
        s = DeadBird.iswgmafter(n) ? " | " : ""
        ls = name.(getleaves(n))  
        color = length(ls) == 1 && ls[1] ∈ ["vvi", "cpa", "bvu"] ? :lightgray : :black
        annotate!(p, ps[n][1], ps[n][2]-0.2, text(s * @sprintf("%4d", q1), :helvetica, color, 8, aln))
        annotate!(p, ps[n][1], ps[n][2]+0.2, text(s * @sprintf("%4d", q2), :helvetica, color, 8, aln))
    end
end
plot(p,left_margin=6Plots.mm, right_margin=33Plots.mm, size=(550,400))

savefig("docs/img/dicots-ancestralstates.pdf")




# Plot ---------------
default(guidefont=10, titlefont=10, title_loc=:left, framestyle=:default)
order = reverse(name.(getleaves(tree)))
p2 = plot(pp3, order=order, taxa=taxa, xscale=:identity, 
          xlim=(1,mx), xticks=(1.5:2:mx+0.5, 0:2:mx))

ps = map(["cqu", "ptr"]) do sp
    rnge = 1:5
    cqu = permutedims(pp3.sims[sp][rnge,:])
    violin(cqu, color=:lightgray, linecolor=:lightgray, xticks=(rnge, rnge .-1))
    y = proportions(data[:,sp])[1:rnge[end]]
    sticks!(rnge, y, color=:black, ms=4)
    scatter!(rnge .+1e-2, y, color=:black, ms=5, title=taxa[sp])
end 
ylabel!.(ps, "\$p_n\$")
xlabel!(ps[2], "\$n\$")
p3 = plot(ps..., ylim=(0,0.7), legend=false, ygrid=false, size=(300,200),
          titlefont=10, titlefontfamily="helvetica oblique", title_loc=:left,
          layout=(2,1))


cs, ls = getcolors(chn3, ftree, "λ")
t1 = plot(ftree, linecolor=cs, lw=2, fontfamily="helvetica oblique", pad=0.7,
          xlim=(-0.0,0.3), title="\$\\lambda\$")
plot_cbar!(t1, exp.(extrema(ls)))

cs, ms = getcolors(chn3, ftree, "μ")
t2 = plot(ftree, linecolor=cs, lw=2, fontfamily="helvetica oblique", pad=0.7,
          xlim=(-0.0,0.3), title="\$\\mu\$")
plot_cbar!(t2, exp.(extrema(ms)))
tp = plot(t1, t2, size=(500,250), colobar=true)

tp = plot(t1, t2, layout=(2,1))
for sp in p2.subplots[1:6]
    xlabel!(sp, "")
end
for sp in [p2.subplots[7:end] ; p3.subplots[2]; tp.subplots[2]]
    sp.attr[:bottom_margin]=3mm
end
for sp in p2.subplots[[2,3,5,6,8,9]]
    ylabel!(sp, "")
end
for sp in p2.subplots[[1,4,7]]
    sp.attr[:left_margin]=3mm
end
pp = plot(p2, p3, tp, size=(1000,400), layout=grid(1,3,widths=[0.65,0.12,0.23]))
for sp in pp.subplots
    sp.attr[:fontfamily_subplot] = "sans-serif"
end
plot(pp)

savefig("docs/img/dicots-brate-wgd.pdf")



function getrates(chain, s)
    map(1:length(chain)) do i 
        makevec(getparams(chain[i]), s)
    end |> x->hcat(x...)
end

function reflected_kde(xs; kwargs...)
    K = kde([xs ; -xs]; kwargs...)
    n = length(K.x) ÷ 2
    K.density = K.density[n+1:end] .* 2
    K.x = K.x[n+1:end]
    return K
end

chn = chn3
d = Dict(id(model1.nodes[k][1][1])=>i for (i,k) in enumerate(wgds))
l = exp.(getrates(chn, "λ"))
m = exp.(getrates(chn, "μ"))
q = getrates(chn, "q")
s =  "| Species | \$q\$ | 95% UI| \$\\lambda\$ | 95% UI | \$\\mu\$ | 95% UI | \$\\log_{10}K\$ |\n"
s *= "| ------- |-----: | --------: | -----------: | ------: | -------: | ------: | --------------: |\n"
for n in getleaves(ftree)
    i = id(n) - 1
    m1 = mean(l[i,:])
    q11, q12 = quantile(l[i,:], [0.025, 0.975])
    m2 = mean(m[i,:])
    q21, q22 = quantile(m[i,:], [0.025, 0.975])
    qs = q[d[id(n)],:]
    m3 = mean(qs)
    q31, q32 = quantile(qs, [0.025, 0.975])
    K = log10(reflected_kde(qs, bandwidth=0.01).density[1])
    sK = K < -2. ? "<-3" : @sprintf "%.1f" K
    s *= @sprintf "| *%s* " name(n)
    s *= @sprintf "| %.1f | (%.1f, %.1f) " m1 q11 q12
    s *= @sprintf "| %.1f | (%.1f, %.1f) " m2 q21 q22
    s *= @sprintf "| %.2f | (%.2f, %.2f) | %s |\n" m3 q31 q32 sK
end
println(s)