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
    @test BirdDad.loglikelihood!(dag, m) ≈ -251.03583571055526
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
@testset "Compare CM algorithm with WGDgc (Cécile Ané)" begin
    tree = readnw("(D:18.03,(C:12.06,(B:7.06,A:7.06):4.99):5.97);")
    X = [2 2 3 4 ; 2 2 3 4]
    s = ["A" "B" "C" "D"]
    dag, bound = CountDAG(X, s, tree)
    r = RatesModel(ConstantDLG(λ=.2, μ=.3, κ=0.0, η=0.9))
    m = PhyloBDP(r, tree, bound)
    ℓ = BirdDad.loglikelihood!(dag, m)
    @test isapprox(ℓ, -19.4707431557829, atol=1e-6)
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

import BirdDad: ConstantDLSC
X, s = readdlm("example/9dicots-f01-100.csv", ',', Int, header=true)
tree = readnw(readline("example/9dicots.nw"))
dag, bound = CountDAG(X, s, tree)
rates = RatesModel(ConstantDLSC(λ=.1, μ=.1, η=1/1.5, m=bound-1))
model = PhyloBDP(rates, tree, bound)
