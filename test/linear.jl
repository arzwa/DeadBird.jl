using Pkg; Pkg.activate(@__DIR__)
using Test, LightGraphs, NewickTree, DeadBird, TransformVariables
using Random, Distributions, CSV
Random.seed!(624)

# NOTE still needs NaN-safe mode enabled sometimes (when κ = 0 for instance)...
const datadir = joinpath(@__DIR__, "../example")
readtree = readnw ∘ readline
 
@testset "CountDAG and Profile" begin
    df = CSV.read(joinpath(datadir, "dicots/9dicots-f01-25.csv"))
    tr = readtree(joinpath(datadir, "dicots/9dicots.nw"))
    dag, bound = CountDAG(df, tr)
    mat, bound = ProfileMatrix(df, tr)
    g = dag.graph
    @test outdegree(g, nv(g)) == length(unique(eachrow(df)))
    @test sum([dag.ndata[i].count for i in outneighbors(g, nv(g))]) == size(df)[1]
    r = RatesModel(ConstantDLG(λ=0.1, μ=.12, κ=0.0, η=0.9))
    m = PhyloBDP(r, tr, bound)
    @test DeadBird.loglikelihood!(dag, m) ≈ -251.0360331682765
    @test DeadBird.loglikelihood!(mat, m) ≈ -251.0360331682765
end

@testset "Profiles" begin
    df = CSV.read(joinpath(datadir, "dicots/9dicots-f01-100.csv"))
    tr = readtree(joinpath(datadir, "dicots/9dicots.nw"))
    dag, bound = CountDAG(df, tr)
    mat, bound = DeadBird.ProfileMatrix(df, tr)
    for i=1:10
        rates = RatesModel(ConstantDLG(λ=.1, μ=.1, κ=0., η=1/1.5), fixed=(:η,:κ))
        model = PhyloBDP(rates(randn(2)), tr, bound)
        l1 = DeadBird.loglikelihood!(mat, model)
        l2 = DeadBird.loglikelihood!(dag, model)
        @test l1 ≈ l2
    end
    for i=1:10
        rates = RatesModel(ConstantDLG(λ=.1, μ=.1, κ=0., η=1/1.5), fixed=())
        model = PhyloBDP(rates(randn(4)), tr, bound)
        l1 = DeadBird.loglikelihood!(mat, model)
        l2 = DeadBird.loglikelihood!(dag, model)
        @test l1 ≈ l2
    end
end

@testset "GDL model..." begin
    i=10
    df = CSV.read(joinpath(datadir, "dicots/9dicots-f01-100.csv"))[i:i,:]
    tr = readtree(joinpath(datadir, "dicots/9dicots.nw"))
    dag, bound = CountDAG(df, tr)
    mat, bound = DeadBird.ProfileMatrix(df, tr)
    rates = RatesModel(ConstantDLG(λ=.1, μ=.1, κ=0., η=1/1.5), fixed=())
    m = PhyloBDP(rates(randn(4)), tr, bound)
    l1 = DeadBird.loglikelihood!(mat, m)
    l2 = DeadBird.loglikelihood!(dag, m)
    @test l1 ≈ l2
end

@testset "GL model..." begin
    df = CSV.read(joinpath(datadir, "dicots/9dicots-f01-100.csv"))
    tr = readtree(joinpath(datadir, "dicots/9dicots.nw"))
    dag, bound = CountDAG(df, tr)
    mat, bound = DeadBird.ProfileMatrix(df, tr)
    m1 = PhyloBDP(RatesModel(ConstantDLG(λ=.0, μ=.2, κ=0.5, η=0.5/0.2),
                             rootprior=:poisson), tr, bound, cond=:none)
    m2 = PhyloBDP(RatesModel(ConstantDLG(λ=1e-7, μ=.2, κ=0.5, η=0.5/0.2), 
                             rootprior=:poisson), tr, bound, cond=:none)
    l1 = DeadBird.loglikelihood!(dag, m1)
    l2 = DeadBird.loglikelihood!(dag, m2)
    @test l1 ≈ l2 atol=1e-5
end

@testset "Gain/no gain" begin
    df = CSV.read(joinpath(datadir, "dicots/9dicots-f01-100.csv"))
    tr = readtree(joinpath(datadir, "dicots/9dicots.nw"))
    dag, bound = CountDAG(df, tr)
    for i=1:10
        matrix, bound = DeadBird.ProfileMatrix(df, tr)
        rates1 = RatesModel(ConstantDLG(λ=.1, μ=.1, κ=0., η=1/1.5), fixed=(:η,:κ))
        model1 = PhyloBDP(rates1(randn(2)), tr, bound)
        rates2 = RatesModel(ConstantDLG(λ=.1, μ=.1, κ=0.2, η=1/1.5), fixed=(:η,))
        model2 = PhyloBDP(rates2(randn(3)), tr, bound)
        l1 = DeadBird.loglikelihood!(matrix, model1)
        l2 = DeadBird.loglikelihood!(dag, model1)
        # some weak testing here...
        @test all([x[1] == -Inf || x[1] == 0. for x in dag.parts])
        @test all([x.ℓ[1][1] == -Inf for x in matrix.profiles])
        l1 = DeadBird.loglikelihood!(matrix, model2)
        l2 = DeadBird.loglikelihood!(dag, model2)
        @test !all([x[1] == -Inf || x[1] == 0. for x in dag.parts]) 
        @test !any([x.ℓ[1][1] == -Inf for x in matrix.profiles])
    end
end

@testset "Drosophila data, DAG vs. matrix" begin
    import DeadBird: loglikelihood!
    df = CSV.read(joinpath(datadir, "drosophila/counts-oib.csv"))
    tr = readtree(joinpath(datadir, "drosophila/tree.nw"))
    dag, bound = CountDAG(df, tr)
    mat, bound = ProfileMatrix(df, tr)
    rates = RatesModel(ConstantDLG(λ=.1, μ=.1, κ=0., η=1/1.5), fixed=(:η,:κ))
    model = PhyloBDP(rates, tr, bound)
    for i=1:10
        r = exp(randn())
        m = model((λ=r, μ=r, η=rand()))
        l1 = loglikelihood!(dag, m)
        l2 = loglikelihood!(mat, m)
        @info l1, l2
        @test isapprox(l1, l2, atol=1e-5)
    end
end

@testset "Drosophila data, DAG vs. matrix vs. WGDgc" begin
    import DeadBird: loglikelihood!
    df = CSV.read(joinpath(datadir, "drosophila/counts-oib.csv"))
    tr = readtree(joinpath(datadir, "drosophila/tree.nw"))
    wgdgc = CSV.read(joinpath(datadir, "drosophila/wgdgc-largest-l1.0-m1.0.csv"))
    rates = RatesModel(ConstantDLG(λ=.1, μ=.1, κ=0., η=1/1.5), fixed=(:η,:κ))
    for i=1:10
        sdf = df[i:i,:]
        dag, bound = CountDAG(sdf, tr)
        mat, bound = ProfileMatrix(sdf, tr)
        model = PhyloBDP(rates, tr, bound)
        m = model((λ=1., μ=1.))
        l1 = loglikelihood!(dag, m)
        l2 = loglikelihood!(mat, m)
        a, b, c = wgdgc[:,i][1:30], dag.parts[end][1:30], mat[1].ℓ[1][1:30]
        @test all(isapprox.(a, c, atol=1e-4))
        @test all(isapprox.(a, b, atol=1e-4))
    end
end

# intensive one
@testset "Gradient issues with large families..." begin
    using ForwardDiff
    df = CSV.read(joinpath(datadir, "drosophila/counts-oib.csv"))
    tr = readtree(joinpath(datadir, "drosophila/tree.nw"))
    for i=1:20
        res = map(1:5) do j
            dag, bound = CountDAG(df[i:i,:], tr)
            mat, bound = ProfileMatrix(df[i:i,:], tr)
            parms = ConstantDLG(λ=.1, μ=.1, κ=0., η=1/1.5)
            rates = RatesModel(parms, fixed=(:η,:κ))
            model = PhyloBDP(rates, tr, bound)
            x = round.(randn(2), digits=2)
            ∇ℓd = ForwardDiff.gradient(x->logpdf(model(x), dag), x)
            ∇ℓp = ForwardDiff.gradient(x->logpdf(model(x), mat), x)
            @test all(isfinite.(∇ℓd))
            @test all(isfinite.(∇ℓp))
            (x, ∇ℓd, ∇ℓp)
        end
        @info "∇" res
    end
end

@testset "Compare CM algorithm with WGDgc (Ané)" begin
    tree = readnw("(D:18.03,(C:12.06,(B:7.06,A:7.06):4.99):5.97);")
    X = [2 2 3 4 ; 2 2 3 4]
    s = ["A" "B" "C" "D"]
    dag, bound = CountDAG(X, s, tree)
    r = RatesModel(ConstantDLG(λ=.2, μ=.3, κ=0.0, η=0.9))
    m = PhyloBDP(r, tree, bound)
    ℓ = DeadBird.loglikelihood!(dag, m)
    @test isapprox(ℓ, -19.624930615416645, atol=1e-6)
    wgdgc = [-Inf, -13.032134, -10.290639, -8.968442, -8.413115,
             -8.380409, -8.78481, -9.592097, -10.801585, -12.448171,
             -14.62676, -17.606982]
    for i=1:length(wgdgc)
        @test isapprox(dag.parts[end][i], wgdgc[i], atol=1e-6)
    end
    root = DeadBird.root(m)
    @test isapprox(root.data.ϵ[2],    log(0.81766934), atol=1e-6)
    @test isapprox(root[1].data.ϵ[1], log(0.93828483), atol=1e-6)
    @test isapprox(root[2].data.ϵ[1], log(0.87145109), atol=1e-6)
end

@testset "Branch rates" begin
    df = CSV.read(joinpath(datadir, "dicots/9dicots-f01-100.csv"))
    tr = readtree(joinpath(datadir, "dicots/9dicots.nw"))
    dag, bound = CountDAG(df, tr)
    n = length(postwalk(tr))
    for i=1:10
        λ, μ, κ = randn(3)
        rates1 = RatesModel(ConstantDLG(λ=exp(λ), μ=exp(μ), κ=exp(κ), η=1/1.5))
        rates2 = RatesModel(DLG(λ=fill(λ, n), μ=fill(μ, n), κ=fill(κ, n), η=1/1.5))
        rates2.params.μ[1] = -Inf  # verify first doesn't matter (root) 
        rates2.params.λ[1] = -Inf  
        rates2.params.κ[1] = -Inf  
        model1 = PhyloBDP(rates1, tr, bound)
        model2 = PhyloBDP(rates2, tr, bound)
        @test logpdf(model1, dag) == logpdf(model2, dag)
    end
end

@testset "MixtureModel" begin
    df = CSV.read(joinpath(datadir, "dicots/9dicots-f01-100.csv"))
    tr = readtree(joinpath(datadir, "dicots/9dicots.nw"))
    dag, bound = CountDAG(df, tr)
    rates = RatesModel(ConstantDLG(λ=.1, μ=.1, κ=0.1, η=1/1.5))
    model = PhyloBDP(rates, tr, bound)
    mixmodel = MixtureModel([model, model])
    @test logpdf(mixmodel, dag) ≈ logpdf(mixmodel.components[1], dag)
    mixmodel = MixtureModel([model(randn(4)) for i=1:4])
    l = logpdf(mixmodel, dag) 
    @info "l" l
    @test -Inf < l < 0.
end

