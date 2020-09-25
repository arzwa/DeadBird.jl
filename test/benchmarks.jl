using BenchmarkTools
import BirdDad: loglikelihood!, cm!

df = CSV.read(joinpath(datadir, "drosophila/counts-oib.csv"))
tr = readtree(joinpath(datadir, "drosophila/tree.nw"))
i = 1000
dag, bound = CountDAG(df[i:i,:], tr)
mat, bound = ProfileMatrix(df[i:i,:], tr)
parms = ConstantDLG(λ=1.1, μ=1.2, κ=0.1, η=1/1.5)
rates = RatesModel(parms, fixed=(:η,:κ))
model = PhyloBDP(rates, tr, bound)

@btime loglikelihood!(mat, model)

@btime loglikelihood!(dag, model)


#= 
family 1


family 1000
julia> @btime cm!(mat[1], model[4], model);
  2.916 μs (54 allocations: 3.56 KiB)

julia> @btime loglikelihood!(mat, model)
  100.771 μs (705 allocations: 67.66 KiB)

julia> @btime loglikelihood!(dag, model)
  97.187 μs (925 allocations: 98.98 KiB)                                          =#

@testset "Csuros-Miklos vs. truncated pruning" begin
    X, s = readdlm("example/9dicots-f01-100.csv", ',', Int, header=true)
    tree = readnw(readline("example/9dicots.nw"))

    # r = exp.(randn(2))
    r = [0.25, 0.2]
    η = rand(Beta(6,2))
    for bound in [10,25,50,100]
        dag, b = CountDAG(X, s, tree)
        rates  = RatesModel(ConstantDLG(λ=r[1], μ=r[2], κ=.0, η=η))
        model1 = PhyloBDP(rates, tree, b)
        ℓ1 = BirdDad.loglikelihood!(dag, model1)
        t1 = @benchmark BirdDad.loglikelihood!(dag, model1)
        @printf "cm: ℓ = %.3f, t = %6.3f, m = %.3f\n" ℓ1 mean(t1.times)/1000 mean(t1.allocs)/1000

        dag_   = BirdDad.nonlineardag(dag, bound)
        rates  = RatesModel(ConstantDLSC(λ=r[1], μ=r[2], μ₁=r[2], η=η, m=bound))
        model2 = PhyloBDP(rates, tree, bound)
        ℓ2 = BirdDad.loglikelihood!(dag_, model2)
        t2 = @benchmark BirdDad.loglikelihood!(dag_, model2)
        @printf "tr: ℓ = %.3f, t = %6.3f, m = %.3f, bound = %d\n\n" ℓ2 mean(t2.times)/1000 mean(t2.allocs)/1000 size(model2.rates.params.Q)[1]
    end
end
