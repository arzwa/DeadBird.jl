using Pkg; Pkg.activate(@__DIR__)
using Test, LightGraphs, NewickTree, BirdDad, TransformVariables
using DelimitedFiles, Random, Distributions
Random.seed!(624)

# NOTE still needs NaN-safe mode enabled sometimes (when κ = 0 for instance)...

@testset "CountDAG" begin
    X, s = readdlm("example/9dicots-f01-25.csv", ',', Int, header=true)
    tree = readnw(readline("example/9dicots.nw"))
    dag, bound = CountDAG(X, s, tree)
    g = dag.graph
    @test outdegree(g, nv(g)) == length(unique(eachrow(X)))
    @test sum([dag.ndata[i].count for i in outneighbors(g, nv(g))]) == size(X)[1]
    r = RatesModel(ConstantDLG(λ=0.1, μ=.12, κ=0.0, η=0.9))
    m = PhyloBDP(r, tree, bound)
    @test BirdDad.loglikelihood!(dag, m) ≈ -251.0360331682765
end

@testset "Profiles" begin
    X, s = readdlm("example/9dicots-f01-100.csv", ',', Int, header=true)
    tree = readnw(readline("example/9dicots.nw"))
    dag, bound = CountDAG(X, s, tree)
    for i=1:10
        matrix, bound = BirdDad.ProfileMatrix(X, s, tree)
        rates = RatesModel(ConstantDLG(λ=.1, μ=.1, κ=0., η=1/1.5), fixed=(:η,:κ))
        model = PhyloBDP(rates(randn(2)), tree, bound)
        l1 = BirdDad.loglikelihood!(matrix, model)
        l2 = BirdDad.loglikelihood!(dag, model)
        @test l1 ≈ l2
    end
end

# This is an important test, comparing to an independent implementation
@testset "Compare CM algorithm with WGDgc (Ané)" begin
    tree = readnw("(D:18.03,(C:12.06,(B:7.06,A:7.06):4.99):5.97);")
    X = [2 2 3 4 ; 2 2 3 4]
    s = ["A" "B" "C" "D"]
    dag, bound = CountDAG(X, s, tree)
    r = RatesModel(ConstantDLG(λ=.2, μ=.3, κ=0.0, η=0.9))
    m = PhyloBDP(r, tree, bound)
    ℓ = BirdDad.loglikelihood!(dag, m)
    # @test isapprox(ℓ, -19.4707431557829, atol=1e-6)  # WGDgc with oneInBothClades
    @test isapprox(ℓ, -19.624930615416645, atol=1e-6)
    wgdgc = [-Inf, -13.032134, -10.290639, -8.968442, -8.413115,
             -8.380409, -8.78481, -9.592097, -10.801585, -12.448171,
             -14.62676, -17.606982]
    for i=1:length(wgdgc)
        @test isapprox(dag.parts[end][i], wgdgc[i], atol=1e-6)
    end
    root = BirdDad.root(m)
    @test isapprox(root.data.ϵ[2], 0.817669336686759, atol=1e-6)
    @test isapprox(root[1].data.ϵ[1], 0.938284827880156, atol=1e-6)
    @test isapprox(root[2].data.ϵ[1], 0.871451090746186, atol=1e-6)
end

@testset "MixtureModel" begin
    X, s = readdlm("example/9dicots-f01-100.csv", ',', Int, header=true)
    tree = readnw(readline("example/9dicots.nw"))
    dag, bound = CountDAG(X, s, tree)
    rates = RatesModel(ConstantDLG(λ=.1, μ=.1, κ=0.1, η=1/1.5))
    model = PhyloBDP(rates, tree, bound)
    mixmodel = MixtureModel([model, model])
    @test logpdf(mixmodel, dag) ≈ logpdf(mixmodel.components[1], dag)
    mixmodel = MixtureModel([model(randn(4)) for i=1:4])
    @test -Inf < logpdf(mixmodel, dag) < 0.
end

@testset "Non-linear models, ConstantDLSC" begin
    import BirdDad: ConstantDLSC
    X, s = readdlm("example/9dicots-f01-100.csv", ',', Int, header=true)
    tree = readnw(readline("example/9dicots.nw"))
    dag, bound = CountDAG(X, s, tree)
    rates = RatesModel(ConstantDLSC(λ=.1, μ=.1, μ₁=0.01, η=1/1.5, m=bound))
    model = PhyloBDP(rates, tree, bound)

    dag = BirdDad.nonlineardag(dag, bound)
    ℓ1 = BirdDad.loglikelihood!(dag, model)

    ps, bound = ProfileMatrix(X, s, tree)
    ps = BirdDad.nonlinearprofile(ps, bound)
    ℓ2 = BirdDad.loglikelihood!(ps, model)
    @test ℓ1 ≈ ℓ2

    # ConstantDLSC with μ₁ == μ should be identical to ConstantDL
    for i=1:10
        r = exp.(randn(2))
        η = rand(Beta(6,2))
        dag, bound = CountDAG(X, s, tree)

        rates = RatesModel(ConstantDLG(λ=r[1], μ=r[2], κ=.0, η=η))
        model1 = PhyloBDP(rates, tree, bound)
        ℓ1 = BirdDad.loglikelihood!(dag, model1)

        dag_ = BirdDad.nonlineardag(dag, 10bound)
        rates= RatesModel(ConstantDLSC(λ=r[1], μ=r[2], μ₁=r[2], η=η, m=10bound))
        model2 = PhyloBDP(rates, tree, 10bound)
        ℓ2 = BirdDad.loglikelihood!(dag_, model2)

        ps, bound = ProfileMatrix(X, s, tree)
        ps = BirdDad.nonlinearprofile(ps, 10bound)
        ℓ3 = BirdDad.loglikelihood!(ps, model2)
        @test ℓ1 ≈ ℓ2 ≈ ℓ3
    end
end

# begin
#     using BenchmarkTools
#     X, s = readdlm("example/9dicots-f01-100.csv", ',', Int, header=true)
#     tree = readnw(readline("example/9dicots.nw"))
#
#     # r = exp.(randn(2))
#     r = [0.25, 0.2]
#     η = rand(Beta(6,2))
#     for bound in [10,25,50,100]
#         dag, b = CountDAG(X, s, tree)
#         rates  = RatesModel(ConstantDLG(λ=r[1], μ=r[2], κ=.0, η=η))
#         model1 = PhyloBDP(rates, tree, b)
#         ℓ1 = BirdDad.loglikelihood!(dag, model1)
#         t1 = @benchmark BirdDad.loglikelihood!(dag, model1)
#         @printf "cm: ℓ = %.3f, t = %6.3f, m = %.3f\n" ℓ1 mean(t1.times)/1000 mean(t1.allocs)/1000
#
#         dag_   = BirdDad.nonlineardag(dag, bound)
#         rates  = RatesModel(ConstantDLSC(λ=r[1], μ=r[2], μ₁=r[2], η=η, m=bound))
#         model2 = PhyloBDP(rates, tree, bound)
#         ℓ2 = BirdDad.loglikelihood!(dag_, model2)
#         t2 = @benchmark BirdDad.loglikelihood!(dag_, model2)
#         @printf "tr: ℓ = %.3f, t = %6.3f, m = %.3f, bound = %d\n\n" ℓ2 mean(t2.times)/1000 mean(t2.allocs)/1000 size(model2.rates.params.Q)[1]
#     end
# end
