# # Benchmarks
# To be used with Literate, to generate a markdown report.
# Note, apparently, Literate cannot handle begin/end blocks?
using BenchmarkTools, BirdDad, CSV, DataFrames, Distributions, NewickTree
import BirdDad: loglikelihood!, cm!

# The current date/time is
using Dates
now()

# Benchmark on Drosophila
datadir = "example/drosophila"
df = CSV.read("$datadir/counts-oib.csv")
tr = readnw(readline("$datadir/tree.nw"))

idx = 1000:1000
mat, bound = ProfileMatrix(df[idx,:], tr)
dag, bound = CountDAG(df[idx,:], tr)
parms = ConstantDLG(λ=1.1, μ=1.2, κ=0.1, η=1/1.5)
rates = RatesModel(parms, fixed=(:η,:κ))
model = PhyloBDP(rates, tr, bound)
bmmat = @benchmark loglikelihood!($(mat), $(model))

benchmarks = map([1000:1000, 10:10, 2000:100:3000, 1:10]) do idx
    dag, bound = CountDAG(df[idx,:], tr)
    mat, bound = ProfileMatrix(df[idx,:], tr)
    parms = ConstantDLG(λ=1.1, μ=1.2, κ=0.1, η=1/1.5)
    rates = RatesModel(parms, fixed=(:η,:κ))
    model = PhyloBDP(rates, tr, bound)
    bmmat = @benchmark loglikelihood!($(mat), $(model))
    bmdag = @benchmark loglikelihood!($(dag), $(model))
    display(bmmat)
    (idx=idx, dag=bmdag, mat=bmmat)
end

# A function to print trials in Literate script
function print_trials(bmarks)
    b = IOBuffer()
    display.(Ref(TextDisplay(b)), bmarks)
    print(String(take!(b)))
end

# For the DAG struct
print_trials(map(x->x.dag, benchmarks))

# For the `ProfileMatrix`
print_trials(map(x->x.mat, benchmarks))

# Comparison of the pruning algorithm on a truncates state space with the CM
# algorithm
#basedir = "example"
#X, s = readdlm("$basedir/9dicots-f01-100.csv", ',', Int, header=true)
#tree = readnw(readline("$basedir/9dicots.nw"))
#r = [0.25, 0.2]
#η = rand(Beta(6,2))
#for bound in [10,25,50,100]
#    dag, b = CountDAG(X, s, tree)
#    rates  = RatesModel(ConstantDLG(λ=r[1], μ=r[2], κ=.0, η=η))
#    model1 = PhyloBDP(rates, tree, b)
#    ℓ1 = BirdDad.loglikelihood!(dag, model1)
#    t1 = @benchmark BirdDad.loglikelihood!($(dag), $(model1))
#    @printf "cm: ℓ = %.3f, t = %6.3f, m = %.3f\n" ℓ1 mean(t1.times)/1000 mean(t1.allocs)/1000
#    dag_   = BirdDad.nonlineardag(dag, bound)
#    rates  = RatesModel(ConstantDLSC(λ=r[1], μ=r[2], μ₁=r[2], η=η, m=bound))
#    model2 = PhyloBDP(rates, tree, bound)
#    ℓ2 = BirdDad.loglikelihood!(dag_, model2)
#    t2 = @benchmark BirdDad.loglikelihood!($(dag_), $(model2))
#    m, a, n = mean(t2.times)/1000 mean(t2.allocs)/1000 size(model2.rates.params.Q)[1]
#    @printf "tr: ℓ = %.3f, t = %6.3f, m = %.3f, bound = %d\n\n" ℓ2 m a n
#end

using Literate #src
Literate.markdown("test/benchmarks.jl", "test",   #src
                  execute=true, documenter=false) #src
