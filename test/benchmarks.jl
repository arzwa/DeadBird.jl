using BenchmarkTools


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
