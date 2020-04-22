using Test, LightGraphs

@testset "Check DAG" begin
    X, s = readdlm("example/9dicots-f01-25.csv", ',', Int, header=true)
    tree = readnw(readline("example/9dicots.nw"))
    dag, bound = CountDAG(X, s, tree)
    g = dag.graph
    @test outdegree(g, nv(g)) == length(unique(eachrow(X)))
    @test sum([dag.ndata[i].count for i in outneighbors(g, nv(g))]) ==
        size(X)[1]
    r = RatesModel(ConstantDLG(λ=0.1, μ=.12, κ=0.0, η=0.9))
    m = PhyloBDP(r, tree, bound)
    @test BirdDad.loglikelihood!(dag, m) ≈ -251.44291115742774
end
