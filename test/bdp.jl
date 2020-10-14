# Arthur Zwaenepoel (2020)
# Tests for BDP internals (numerical accuracy etc.)
using Test

@testset "Basics" begin
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
