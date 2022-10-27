
using DeadBird.YuleSimonDistribution
default(gridstyle=:dot)
theme(:wong2)

dir = "/home/arzwa/research/gene-family-evolution/data/angiosperm/"
df = CSV.read("$dir/angiosperm.N0.tsv", DataFrame)
tr = readnw("((atr,vvi),mpo);")
data, cols = DeadBird.getcountsof(df, tr) 

xs = map(sp->counts(filter(x-> x > zero(x), data[:,sp])), [:atr, :vvi, :mpo])

function getparams(x)
    vars = x.value.axes[2]
    (; [var=>x for (var,x) in zip(vars, vec(x.value.data))]...)
end
bound01(η) = η <= zero(η) ? zero(η) + 1e-16 : η >= one(η) ? one(η) - 1e-16 : η

@model gfit(xs) = begin
    η ~ Beta()
    M = Geometric(bound01(η))
    for (k,x) in enumerate(xs)
        Turing.@addlogprob!(x*logpdf(M,k-1))
    end
end
function gsim(chain, Y)
    Ys = map(i->rand(Geometric(get(chain[i], :η).η[1]), sum(Y)) .+ 1,
             1:length(chain))
    Ys = mapreduce(x->proportions(x, 1:length(Y)), hcat, Ys)
    DeadBird.quantiles(Ys)
end

@model bgfit(xs) = begin
    η ~ Beta()
    ζ ~ Turing.FlatPos(0.)
    M = ShiftedBetaGeometric(η, ζ)
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


@model ysfit(xs) = begin
    ρ ~ Turing.FlatPos(0.)
    M = YuleSimonDistribution.YuleSimon(ρ)
    for (k,x) in enumerate(xs)
        Turing.@addlogprob!(x*logpdf(M,k))
    end
end
function yssim(chain, Y)
    Ys = map(i->rand(YuleSimonDistribution.YuleSimon(get(chain[i], :ρ).ρ[1]),
                     sum(Y)), 1:length(chain))
    Ys = mapreduce(x->proportions(x, 1:length(Y)), hcat, Ys)
    DeadBird.quantiles(Ys)
end


models = [(gfit, gsim), (ysfit, yssim), (bgfit, bgsim)]
ps = map(xs) do X
    chns = map(models) do (model, simf)
        chn = sample(model(X), NUTS(), 500)
        pps = simf(chn, X)
        chn, pps
    end
    X, chns
end

labels = ["Geometric", "Yule-Simon", "Beta-geometric"]
titles = ["Amborella", "Vitis", "Marchantia"]
xx = map(zip(titles, ps)) do (sp, (X, chns))
    y = log10.(X ./sum(X))
    p = scatter(1.5:length(y) + 0.5, y, xscale=:log10, color=:black, label="",
                title=sp, xlabel="\$n\$", ylabel="\$\\log_{10}f_n\$")
    i = 1
    map(chns) do (chn, Ys)
        stepplot!(Ys[:,1], color=i, ribbon=(Ys[:,2], Ys[:,3]), fillalpha=0.2,
                  label=labels[i])
        i += 1
    end
    p
end
plot(xx..., size=(800,230), layout=(1,3), xlim=(1,100), legend=:topright,
     fg_legend=:transparent, titlefont=9, titlefontfamily="helvetica oblique",
     title_loc=:left, bottom_margin=7mm, left_margin=3mm)

savefig("docs/img/paranome-fit-angiosperm.pdf")
