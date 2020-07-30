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

# non-linear models
function setW!(n::ModelNode{T}, rates) where T
    isroot(n) && return
    Q = getQ(rates.params, n)
    n.data.W .= exp(Q*distance(n))
end

# NOTE: for non-linear models the pgf formulation can't be used
# (because ℙ{extinction} = F(0) is only valid for processes
# with the branching property)
# function setϵ!(n::ModelNode{T}, rates) where T
#     isleaf(n) && return  # XXX or should we set ϵ to 0.? [it should always be]
#     setϵ!(n, 2, one(T))
#     for c in children(n)
#         𝑃 = c.data.W
#         ϵc = extp(θc.λ, θc.μ, distance(c), getϵ(c, 2))
#         setϵ!(c, 1, ϵc)
#         setϵ!(n, 2, probify(getϵ(n, 2) * ϵc))
#     end
#     # end
# end

function loglikelihood!(dag::CountDAG, model)
    for level in dag.levels  # parallelism possible within levels
        # Threads.@threads for n in level
        for n in level
            LightGraphs.outdegree(dag.graph, n) == 0 && continue
            prune!(dag, n, model)
        end
    end
    ℓ = acclogpdf(dag, model) - dag.nfam*conditionfactor(model)
    isfinite(ℓ) ? ℓ : -Inf
end

@inline function prune!(dag::CountDAG{T}, n, model) where T
    @unpack ndata, parts, graph = dag
    parts[n] .= zero(T)
    # loop over childnodes
    for c in outneighbors(graph, n)
        𝑃 = model[ndata[c].snode].data.W
        parts[n] .+= log.(𝑃 * exp.(parts[c]))
    end
end

# nonlinear models
function loglikelihood!(p::Profile, model, condition=true)
    @unpack η = getθ(model.rates, root(model))
    ϵ = log(probify(getϵ(root(model), 2)))
    for n in model.order
        prune!(p, n, model)
    end
    ℓ = ∫rootgeometric(p.ℓ[1], η, ϵ)
    if condition
        ℓ -= conditionfactor(model)
    end
    isfinite(ℓ) ? ℓ : -Inf
end

@inline function prune!(p::Profile{T}, n, model) where T
    @unpack x, ℓ = p
    if isleaf(n)
        ℓ[id(n)][x[id(n)]+1] = 0.
        return
    end
    ℓ[id(n)] .= zero(T)
    for c in children(n)
        𝑃 = model[id(c)].data.W
        ℓ[id(n)] .+= log.(𝑃 * exp.(ℓ[id(c)]))
    end
end
