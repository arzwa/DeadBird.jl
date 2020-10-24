
# not supported anymore?
@testset "Non-linear models, ConstantDLSC" begin
    import DeadBird: ConstantDLSC
    df = CSV.read(joinpath(datadir, "dicots/9dicots-f01-100.csv"))
    tr = readtree(joinpath(datadir, "dicots/9dicots.nw"))
    dag, bound = CountDAG(df, tr)
    rates = RatesModel(ConstantDLSC(λ=.1, μ=.1, μ₁=0.01, η=1/1.5, m=bound))
    model = PhyloBDP(rates, tr, bound)

    dag = DeadBird.nonlineardag(dag, bound)
    ℓ1 = DeadBird.loglikelihood!(dag, model)

    ps, bound = ProfileMatrix(df, tr)
    ps = DeadBird.nonlinearprofile(ps, bound)
    ℓ2 = DeadBird.loglikelihood!(ps, model)
    @test ℓ1 ≈ ℓ2

    # ConstantDLSC with μ₁ == μ should be identical to ConstantDL
    for i=1:10
        r = exp.(randn(2))
        η = rand(Beta(6,2))
        dag, bound = CountDAG(df, tr)

        rates = RatesModel(ConstantDLG(λ=r[1], μ=r[2], κ=.0, η=η))
        model1 = PhyloBDP(rates, tr, bound)
        ℓ1 = DeadBird.loglikelihood!(dag, model1)

        dag_ = DeadBird.nonlineardag(dag, 10bound)
        rates= RatesModel(ConstantDLSC(λ=r[1], μ=r[2], μ₁=r[2], η=η, m=10bound))
        model2 = PhyloBDP(rates, tr, 10bound)
        ℓ2 = DeadBird.loglikelihood!(dag_, model2)

        ps, bound = ProfileMatrix(df, tr)
        ps = DeadBird.nonlinearprofile(ps, 10bound)
        ℓ3 = DeadBird.loglikelihood!(ps, model2)
        @test ℓ1 ≈ ℓ2 ≈ ℓ3
    end
end


