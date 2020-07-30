using Pkg; Pkg.activate(@__DIR__)
using Test, LightGraphs, NewickTree, BirdDad, TransformVariables
using DelimitedFiles, Optim, LineSearches, Random
Random.seed!(624)

# NOTE still needs NaN-safe mode enabled sometimes (when κ = 0 for instance)...
# @testset "ML for constant rates" begin
#     X, s = readdlm("example/9dicots-f01-1000.csv", ',', Int, header=true)
#     tree = readnw(readline("example/9dicots.nw"))
#     dag, bound = CountDAG(X, s, tree)
#     rates = RatesModel(ConstantDLG(λ=.1, μ=.1, κ=1e-15, η=1/1.5), fixed=(:η, :κ))
#     model = PhyloBDP(rates, tree, bound)
#     f, ∇f = mle_problem(dag, model)
#     @test ∇f(zeros(2), randn(2))[2] ≈ 16.7390819589
#     timed = @timed out = optimize(f, ∇f, randn(2), BFGS())
#     # @test timed[2] < 1.  # almost order of magnitude faster than WGDgc
#     t = transform(model.rates.trans, out.minimizer)
#     @test isapprox(t.λ, 0.3406234, atol=1e-6)  # 0.3406232  WGDgc
#     @test isapprox(t.μ, 0.2178058, atol=1e-6)  # 0.2178061
# end
#
# @testset "Gamma mixture" begin
#     X, s = readdlm("example/9dicots-f01-1000.csv", ',', Int, header=true)
#     tree = readnw(readline("example/9dicots.nw"))
#     dag, bound = CountDAG(X, s, tree)
#
#     # K = 1 should be identical to no mixture
#     rates1 = RatesModel(ConstantDLG(λ=.1, μ=.1, κ=1e-9, η=1/1.5))
#     rates2 = RatesModel(GammaMixture(
#         ConstantDLG(λ=.1, μ=.1, κ=1e-9, η=1/1.5), 1, α=exp(randn())))
#     model1 = PhyloBDP(rates1, tree, bound)
#     model2 = PhyloBDP(rates2, tree, bound)
#     BirdDad.loglikelihood!(dag, model1) ≈
#         BirdDad.loglikelihood!(dag, model2)
#
#     for α=[0.1, 1., 10], K=2:8
#         rates = RatesModel(
#             GammaMixture(ConstantDLG(λ=.1, μ=.1, κ=1e-9, η=1/1.5), K, α=α),
#             fixed=(:η, :κ))
#         @test sum(rates.params.rrates) ≈ K
#         model = PhyloBDP(rates, tree, bound)
#         ℓ = BirdDad.loglikelihood!(dag, model)
#         @test -12800 < ℓ < -12000
#     end
# end
#
# @testset "ML estimation for Gamma mixture" begin
#     X, s = readdlm("example/9dicots-f01-100.csv", ',', Int, header=true)
#     tree = readnw(readline("example/9dicots.nw"))
#     dag, bound = CountDAG(X, s, tree)
#
#     rates = RatesModel(
#         GammaMixture(ConstantDLG(λ=.1, μ=.1, κ=0., η=1/1.5), 2, α=0.1),
#         fixed=(:η, :κ, :α))
#     model = PhyloBDP(rates, tree, bound)
#     f, ∇f = mle_problem(dag, model)
#     out = optimize(f, ∇f, randn(2),
#         LBFGS(linesearch=LineSearches.BackTracking()))
#     @show out.minimizer
#     # @test out.minimizer[1] ≈ -1.8817139345050482
#     # @test out.minimizer[2] ≈ -2.2428853925667456
#
#     rates = RatesModel(
#         GammaMixture(ConstantDLG(λ=.1, μ=.1, κ=0., η=1/1.5), 2, α=1.),
#         fixed=(:η, :κ))
#     model = PhyloBDP(rates, tree, bound)
#     f, ∇f = mle_problem(dag, model)
#     out = optimize(f, ∇f, randn(3),
#         LBFGS(linesearch=LineSearches.BackTracking()))
#     @show out.minimizer
# end
