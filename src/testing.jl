using NewickTree
using Parameters
using StatsFuns
using Test, BenchmarkTools

t = readnw(readline("example/9dicots.nw"))
r = ConstantDLG(λ=1.0, μ=1.2, η=0.9)
m = PhyloBDP(r, t, 13)
x = [12, 6, 1, 5, 2, 1, 1, 3, 1, 2, 6, 3, 1, 2, 3, 1, 2]
L = fill(-Inf, (13,17))
compute_conditionals!(L, x, m)

shouldbe = [-Inf, -9.756251, -11.236213, -18.206932, -26.424245, -35.508428,
            -45.336718, -55.847512, -67.019011, -78.884863, -91.455586,
            -104.807965, -119.2536]
@testset "Still OK?" begin
    for i=1:length(shouldbe) @test L[i,1] ≈ shouldbe[i] atol=1e-6 end
end
@btime cm!($L, $x, $m.nodes[4])
@btime compute_conditionals!($L, $x, $m)

t = readnw("((A:1,B:1):1,C:2);")
r = ConstantDLG(λ=1.0, μ=1.2, η=0.9)
x1 = [3, 2, 1, 1, 1]
L1 = fill(-Inf, (5,5))
x2 = [4, 2, 1, 1, 2]
L2 = fill(-Inf, (5,5))
x3 = [4, 2, 2, 0, 2]
L3 = fill(-Inf, (5,5))
m = PhyloBDP(r, t, 5)
compute_conditionals!(L1, x1, m)
compute_conditionals!(L2, x2, m)
compute_conditionals!(L3, x3, m);
