# Truncated birth-death process models
# with computations using a classical pruning algorithm.
# The likelihood for DL, DLG and *WGD models can be computed using the CM algorithm
# but for more complicated models, we often need to truncate the state space
# and perform matrix exponentiation.
# This is because the CM algorithm relies on the underlying process being
# a proper branching process,
# i.e. where evolution of each particle is independent of the other particles.
# The `PhyloBDP` struct can be used directly for these purposes,
# we have the W field in the `ModelNode` object to store
# the transition probability matrices.
# So we could just dispatch on the `RatesModel` to know whether we need
# the truncated state space models or whether we can use the CM algorithm.
"""
    ConstantDLSC{T}

Constant rates duplication-loss model with different loss rates
when in single copy state.
"""
struct ConstantDLSC{T} <: Params{T}
    λ ::T
    μ ::T
    μ₁::T
    η ::T
    m ::Int   # truncation bound
    Q ::Matrix{T}

    function ConstantDLSC(λ::T, μ::T, μ₁::T, η::T, m::Int) where T
        p = new{T}(λ, μ, μ₁, η, m, Matrix{T}(undef, m+1, m+1))
        setratematrix!(p)
        return p
    end
end

ConstantDLSC(; λ=0.1, μ=0.1, μ₁=0.01, η=0.66, m=10) =
    ConstantDLSC(promote(λ, μ, μ₁, η)..., m)

Base.show(io::IO, m::ConstantDLSC) = write(io, "ConstantDLSC(\n λ  = $(m.λ),",
    "\n μ  = $(m.μ),\n μ₁ = $(m.μ₁),\n η  = $(m.η),\n m  = $(m.m))")

getθ(m::ConstantDLSC, node) = m
getQ(m::ConstantDLSC, node) = m.Q
trans(::ConstantDLSC) = (λ=asℝ₊, μ=asℝ₊, μ₁=asℝ₊, η=as𝕀)
(::ConstantDLSC)(θ) = ConstantDLSC(; λ=θ.λ, μ=θ.μ, μ₁=θ.μ₁, η=θ.η)

function setratematrix!(p::ConstantDLSC)
    @unpack λ, μ, μ₁, η, m = p
    μs = [μ₁ ; μ .* collect(2:m)]
    λs = λ .* collect(0:(m-1))
    ds = vcat(0., -λs[2:end] .- μs[1:end-1], -μs[end])
    p.Q .= Matrix(BandedMatrix(-1=>μs, 1=>λs, 0=>ds))
end


function loglikelihood!(dag::CountDAG, model::PhyloBDP{T}) where {T}
    for level in dag.levels  # parallelism possible within levels
        Threads.@threads for n in level
            prune!(dag, n, model)
        end
    end
    ℓ = acclogpdf(dag, model) - dag.nfam*conditionfactor(model)
    isfinite(ℓ) ? ℓ : -Inf
end
