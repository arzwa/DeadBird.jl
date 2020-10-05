using Pkg; Pkg.activate(@__DIR__)
using Test, LightGraphs, NewickTree, BirdDad, TransformVariables
using Random, Distributions, CSV
Random.seed!(624)

# NOTE still needs NaN-safe mode enabled sometimes (when κ = 0 for instance)...
const datadir = joinpath(@__DIR__, "../example")
readtree = readnw ∘ readline

@testset "CountDAG and Profile" begin
    df = CSV.read(joinpath(datadir, "9dicots-f01-25.csv"))
    tr = readtree(joinpath(datadir, "9dicots.nw"))
    dag, bound = CountDAG(df, tr)
    mat, bound = ProfileMatrix(df, tr)
    g = dag.graph
    @test outdegree(g, nv(g)) == length(unique(eachrow(df)))
    @test sum([dag.ndata[i].count for i in outneighbors(g, nv(g))]) == size(df)[1]
    r = RatesModel(ConstantDLG(λ=0.1, μ=.12, κ=0.0, η=0.9))
    m = PhyloBDP(r, tr, bound)
    @test BirdDad.loglikelihood!(dag, m) ≈ -251.0360331682765
    @test BirdDad.loglikelihood!(mat, m) ≈ -251.0360331682765
end

@testset "Profiles" begin
    df = CSV.read(joinpath(datadir, "9dicots-f01-100.csv"))
    tr = readtree(joinpath(datadir, "9dicots.nw"))
    dag, bound = CountDAG(df, tr)
    for i=1:10
        matrix, bound = BirdDad.ProfileMatrix(df, tr)
        rates = RatesModel(ConstantDLG(λ=.1, μ=.1, κ=0., η=1/1.5), fixed=(:η,:κ))
        model = PhyloBDP(rates(randn(2)), tr, bound)
        l1 = BirdDad.loglikelihood!(matrix, model)
        l2 = BirdDad.loglikelihood!(dag, model)
        @test l1 ≈ l2
    end
end

@testset "Drosophila data, DAG vs. matrix" begin
    import BirdDad: loglikelihood!
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
    import BirdDad: loglikelihood!
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
    ℓ = BirdDad.loglikelihood!(dag, m)
    @test isapprox(ℓ, -19.624930615416645, atol=1e-6)
    wgdgc = [-Inf, -13.032134, -10.290639, -8.968442, -8.413115,
             -8.380409, -8.78481, -9.592097, -10.801585, -12.448171,
             -14.62676, -17.606982]
    for i=1:length(wgdgc)
        @test isapprox(dag.parts[end][i], wgdgc[i], atol=1e-6)
    end
    root = BirdDad.root(m)
    @test isapprox(root.data.ϵ[2],    log(0.81766934), atol=1e-6)
    @test isapprox(root[1].data.ϵ[1], log(0.93828483), atol=1e-6)
    @test isapprox(root[2].data.ϵ[1], log(0.87145109), atol=1e-6)
end

@testset "MixtureModel" begin
    df = CSV.read(joinpath(datadir, "9dicots-f01-100.csv"))
    tr = readtree(joinpath(datadir, "9dicots.nw"))
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

# not supported anymore?
@testset "Non-linear models, ConstantDLSC" begin
    import BirdDad: ConstantDLSC
    df = CSV.read(joinpath(datadir, "9dicots-f01-100.csv"))
    tr = readtree(joinpath(datadir, "9dicots.nw"))
    dag, bound = CountDAG(df, tr)
    rates = RatesModel(ConstantDLSC(λ=.1, μ=.1, μ₁=0.01, η=1/1.5, m=bound))
    model = PhyloBDP(rates, tr, bound)

    dag = BirdDad.nonlineardag(dag, bound)
    ℓ1 = BirdDad.loglikelihood!(dag, model)

    ps, bound = ProfileMatrix(df, tr)
    ps = BirdDad.nonlinearprofile(ps, bound)
    ℓ2 = BirdDad.loglikelihood!(ps, model)
    @test ℓ1 ≈ ℓ2

    # ConstantDLSC with μ₁ == μ should be identical to ConstantDL
    for i=1:10
        r = exp.(randn(2))
        η = rand(Beta(6,2))
        dag, bound = CountDAG(df, tr)

        rates = RatesModel(ConstantDLG(λ=r[1], μ=r[2], κ=.0, η=η))
        model1 = PhyloBDP(rates, tr, bound)
        ℓ1 = BirdDad.loglikelihood!(dag, model1)

        dag_ = BirdDad.nonlineardag(dag, 10bound)
        rates= RatesModel(ConstantDLSC(λ=r[1], μ=r[2], μ₁=r[2], η=η, m=10bound))
        model2 = PhyloBDP(rates, tr, 10bound)
        ℓ2 = BirdDad.loglikelihood!(dag_, model2)

        ps, bound = ProfileMatrix(df, tr)
        ps = BirdDad.nonlinearprofile(ps, 10bound)
        ℓ3 = BirdDad.loglikelihood!(ps, model2)
        @test ℓ1 ≈ ℓ2 ≈ ℓ3
    end
end