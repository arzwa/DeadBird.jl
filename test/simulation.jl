using Pkg; Pkg.activate(@__DIR__)
using Test, LightGraphs, NewickTree, DeadBird, TransformVariables
using Random, Distributions, CSV
using Turing
Random.seed!(624)

const tree = readnw("((A:1,B:1):0.5,(C:0.3,D:0.3):1.2);")

@testset "Constant rates" begin
    θ = ConstantDLG(λ=0.5, μ=0.4, κ=0., η=0.7)
    m = PhyloBDP(RatesModel(θ), tree, 1)
    X = simulate(m, 100)[:,1:end-2]
    Y, bound = CountDAG(X, tree)
    m = m(bound)
    @model tmodel(Y, model) = begin 
        λ ~ FlatPos(0.)
        μ ~ FlatPos(0.)
        η ~ Beta(3,1)
        Y ~ model((λ=λ,μ=μ,η=η,κ=0.))
    end
    chain = sample(tmodel(Y, m), NUTS(), 500)
    qs = quantile(chain, q=[0.025, 0.975])
    for s in [:η, :λ, :μ]
        @test qs[s].nt[2][1] < getfield(θ, s) < qs[s].nt[3][1]
    end
end

@testset "Constant rates with gain" begin
    θ = ConstantDLG(λ=0.5, μ=0.4, κ=0.3, η=0.7)
    m = PhyloBDP(RatesModel(θ, rootprior=:geometric), tree, 1, cond=:none)
    X = simulate(m, 500)[:,1:end-2]
    Y, bound = CountDAG(X, tree)
    m = m(bound)
    @model tmodel(Y, model) = begin 
        λ ~ FlatPos(0.)
        μ ~ FlatPos(0.)
        κ ~ FlatPos(0.)
        η ~ Beta(3,1)
        Y ~ model((λ=λ,μ=μ,η=η,κ=κ))
    end
    chain = sample(tmodel(Y, m), NUTS(), 500)
    qs = quantile(chain, q=[0.025, 0.975])
    for s in [:η, :λ, :μ, :κ]
        @test qs[s].nt[2][1] < getfield(θ, s) < qs[s].nt[3][1]
    end
end

@testset "Constant rates with gain, core model" begin
    θ = (λ=0.5, η=2/3)
    p = ConstantDLG(λ=θ.λ, μ=θ.λ/(1-θ.η), κ=θ.λ, η=θ.η)
    m = PhyloBDP(RatesModel(p, rootprior=:geometric), tree, 1, cond=:none)
    X = simulate(m, 500)[:,1:end-2]
    Y1, bound = CountDAG(X, tree)
    Y2, bound = ProfileMatrix(X, tree)
    m = m(bound)
    @model tmodel(Y, model) = begin 
        λ ~ FlatPos(0.)
        η ~ Beta()
        Y ~ model((λ=λ, μ=λ/(1-η), κ=λ, η=η))
    end
    chain1 = sample(tmodel(Y1, m), NUTS(), 100)
    chain2 = sample(tmodel(Y2, m), NUTS(), 100)
    qs1 = quantile(chain1, q=[0.025, 0.975])
    qs2 = quantile(chain2, q=[0.025, 0.975])
    for s in [:η, :λ], qs in [qs1, qs2]
        x = getfield(θ, s) 
        @info "true value and 95% uncertainty interval" x qs[s]
        @test qs[s].nt[2][1] < x < qs[s].nt[3][1]
    end
end
