using DeadBird, Test 
using LightGraphs, NewickTree, Random, Distributions, CSV, DataFrames, ForwardDiff
using DeadBird: loglikelihood!
Random.seed!(624)

# XXX do we still need NaN-safe mode in ForwardDiff?
const datadir = joinpath(@__DIR__, "../example")
readtree = readnw ∘ readline

@testset "DeadBird, linear BDIPs" begin
    df = CSV.read(joinpath(datadir, "dicots/9dicots-f01-25.csv"), DataFrame)
    tr = readtree(joinpath(datadir, "dicots/9dicots.nw"))
    dag, bound = CountDAG(df, tr)
    mat, bound = ProfileMatrix(df, tr)
    
    @testset "Numerical stability/robustness" begin
        # Numerical stability of ϕ and ψ
        import DeadBird: getϕψ, getϕψ′, extp
        for l=-15:2:15, m=-15:2:15, t=-10:2:10
            λ = exp(l); μ = exp(m); t = exp(t)
            ϕ, ψ = getϕψ(t, λ, μ)
            @test zero(ϕ) <= ϕ <= one(ϕ)
            @test zero(ϕ) <= ψ <= one(ϕ)
            for e=0:2:20
                ϵ = extp(t, λ, μ, exp10(-e))
                @test zero(ϵ) <= ϵ <= one(ϵ)
                ϕ′, ψ′ = getϕψ′(ϕ, ψ, ϵ)
                @test zero(ϕ) <= ϕ′ <= one(ϕ)
                @test zero(ϕ) <= ψ′ <= one(ϕ)
            end
            @test extp(t, λ, μ, 0.) ≈ ϕ
        end
    end
    
    @testset "CountDAG and Profile, default linear BDP" begin
        g = dag.graph
        @test outdegree(g, nv(g)) == length(unique(eachrow(df)))
        @test sum([dag.ndata[i].count for i in outneighbors(g, nv(g))]) == size(df)[1]
        r = RatesModel(ConstantDLG(λ=0.1, μ=.12, κ=0.0, η=0.9))
        m = PhyloBDP(r, tr, bound)
        @test loglikelihood!(dag, m) ≈ -251.0360331682765
        @test loglikelihood!(mat, m) ≈ -251.0360331682765
    end
    
    @testset "Gradient linear BDIP" begin
        r = RatesModel(ConstantDLG(λ=0.1, μ=.12, κ=0.11, η=0.9))
        m = PhyloBDP(r, tr, bound)
        y = [0.1, 0.12, 0.11, 0.90]
        g1 = ForwardDiff.gradient(x->logpdf(m((λ=x[1], μ=x[2], κ=x[3], η=x[4])), dag), y)
        g2 = ForwardDiff.gradient(x->logpdf(m((λ=x[1], μ=x[2], κ=x[3], η=x[4])), mat), y)
        g_ = [98.9170420717962, -7.469684450808845, 63.49317519767404, 3.000649823968922]
        @test g1 ≈ g2 ≈ g_
    end

    @testset "CountDAG and Profile, DLG" begin
        for i=1:10
            rates = RatesModel(ConstantDLG(λ=.1, μ=.1, κ=0.1, η=1/1.5), rootprior=:geometric)
            model = PhyloBDP(rates(randn(4)), tr, bound, cond=:none)
            l1 = loglikelihood!(mat, model)
            l2 = loglikelihood!(dag, model)
            @test isfinite(l1)
            @test l1 ≈ l2
        end
    end

    @testset "GL model" begin
        r1 = RatesModel(ConstantDLG(λ=.0,   μ=.2, κ=0.5, η=0.5/0.2), rootprior=:poisson)
        r2 = RatesModel(ConstantDLG(λ=1e-7, μ=.2, κ=0.5, η=0.5/0.2), rootprior=:poisson)
        m1 = PhyloBDP(r1, tr, bound, cond=:none)
        m2 = PhyloBDP(r2, tr, bound, cond=:none)
        l1 = loglikelihood!(dag, m1)
        l2 = loglikelihood!(dag, m2)
        @test l1 ≈ l2 atol=1e-5
    end

    @testset "Gain/no gain" begin
        for i=1:10
            λ, μ, κ = exp.(randn(3))
            r1 = RatesModel(ConstantDLG(λ=λ, μ=μ, κ=0.0, η=1/1.5))
            r2 = RatesModel(ConstantDLG(λ=λ, μ=μ, κ=κ, η=1/1.5))
            m1 = PhyloBDP(r1, tr, bound)
            m2 = PhyloBDP(r2, tr, bound)
            l1 = loglikelihood!(mat, m1)
            l2 = loglikelihood!(dag, m1)
            # some weak testing here...
            @test all([x[1] == -Inf || x[1] == 0. for x in dag.parts])
            @test all([x.ℓ[1][1] == -Inf for x in mat.profiles])
            l1 = loglikelihood!(mat, m2)
            l2 = loglikelihood!(dag, m2)
            @test !all([x[1] == -Inf || x[1] == 0. for x in dag.parts]) 
            @test !any([x.ℓ[1][1] == -Inf for x in mat.profiles])
        end
    end
    
    @testset "Compare CM algorithm with WGDgc (Rabier, Ta, Ané)" begin
        tree = readnw("(D:18.03,(C:12.06,(B:7.06,A:7.06):4.99):5.97);")
        X = [2 2 3 4 ; 2 2 3 4]
        s = ["A" "B" "C" "D"]
        dag, bound = CountDAG(X, s, tree)
        r = RatesModel(ConstantDLG(λ=.2, μ=.3, κ=0.0, η=0.9))
        m = PhyloBDP(r, tree, bound)
        ℓ = loglikelihood!(dag, m)
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
        dag, bound = CountDAG(df, tr)
        rates = RatesModel(ConstantDLG(λ=.1, μ=.1, κ=0.1, η=1/1.5))
        model = PhyloBDP(rates, tr, bound)
        mixmodel = MixtureModel([model, model])
        @test logpdf(mixmodel, dag) ≈ logpdf(mixmodel.components[1], dag)
        mixmodel = MixtureModel([model(randn(4)) for i=1:4])
        l = logpdf(mixmodel, dag) 
        #@info "l" l
        @test -Inf < l < 0.
    end

    @testset "Drosophila data" begin
        dd = joinpath(datadir, "drosophila")
        df = CSV.read(joinpath(dd, "counts-oib.csv"), DataFrame)
        tr = readtree(joinpath(dd, "tree.nw"))
        wgdgc = CSV.read(joinpath(dd, "wgdgc-largest-l1.0-m1.0.csv"), DataFrame)
        
        @testset "Drosophila data, DAG vs. matrix vs. WGDgc" begin
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

        # intensive one, this was a source of numerical issues previously
        @testset "Gradients for large families" begin
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
                #@info "∇" res
            end
        end
    end
end

