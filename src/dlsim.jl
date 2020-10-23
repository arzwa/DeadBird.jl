# Arthur Zwaenepoel (2020)
# Simulate profiles directly for *linear* BDPs This does not simulate trees,
# but only counts. Probably should be the `rand` function. 
# For posterior predictive simulations, this better be fast!
# see: https://docs.julialang.org/en/v1/stdlib/Random/#An-optimized-sampler-with-pre-computed-data
# A bit tricky that the Distributions.jl and Random interfaces for sampling are
# different. I use the Random API here...
#Distributions.rand(rng::AbstractRNG, m::LPhyloBDP) = rand(rng, Sampler(rng, m)) 
#Distributions.rand(rng::AbstractRNG, m::LPhyloBDP, d::Int64) = rand(rng, Sampler(rng, m), d) 
#
#function Random.Sampler(::Type{<:AbstractRNG}, m::LPhyloBDP, ::Random.Repetition)
#    idx  = getleafindex(m)
#    f    = getcondition(m, idx)
#    ks   = first.(sort(collect(idx), by=x->last(x)))
#    cols = vcat(ks..., ["rejected", "extinct"])
#    data = (idx=idx, f=f, cols=cols)
#    Random.SamplerSimple(m, data) 
#end
#
#Random.eltype(::Type{<:LPhyloBDP}) = Vector{Int}
#
#function Random.rand(rng::AbstractRNG, sp::Random.SamplerSimple{<:LPhyloBDP}) 
#    @unpack self, data = sp
#    @unpack idx, f, cols = data
#    x = simulate_profile(rng, self, idx, f)  # TODO: pass rng...
#end
#
#function Random.rand(rng::AbstractRNG, sp, d::Integer) 
#    x = rand(rng, sp, Dims((d,)))
#    X = hcat(x...) |> permutedims
#    DataFrame(X, Symbol.(sp.data.cols))
#end
#
#Random.rand(sp, d::Integer) = rand(Random.default_rng(), sp, d)
#
#Random.rand(rng::AbstractRNG, m::ModelArray) = vcat([rand(rng, x, 1) for x in m.models]...)

# for several reasons it seems more straightforward to not use the Random API
# (but we do use a similar approach).

struct ProfileSim{T}
    model::T
    idx  ::Dict
    cond ::Function
    cols ::Vector{Symbol}
    function ProfileSim(m::T) where T 
        idx  = getleafindex(m)
        cond = getcondition(m, idx)
        ks   = first.(sort(collect(idx), by=x->last(x)))
        cols = vcat(ks..., ["rejected", "extinct"])
        new{T}(m, idx, cond, Symbol.(cols))
    end
end

"""
    simulate(m::ModelArray)
    simulate(m::MixtureModel, n)
    simulate(m::PhyloBDP, n)

Simulate a set of random profiles from a phylogenetic birth-death model.

# Example
```julia-repl
julia> x = DeadBird.example_data();

julia> simulate(x.model, 5)
5×5 DataFrame
│ Row │ A     │ B     │ C     │ rejected │ extinct │
│     │ Int64 │ Int64 │ Int64 │ Int64    │ Int64   │
├─────┼───────┼───────┼───────┼──────────┼─────────┤
│ 1   │ 1     │ 1     │ 1     │ 0        │ 0       │
│ 2   │ 1     │ 1     │ 1     │ 0        │ 0       │
│ 3   │ 2     │ 2     │ 2     │ 0        │ 0       │
│ 4   │ 0     │ 1     │ 1     │ 1        │ 1       │
│ 5   │ 1     │ 1     │ 1     │ 0        │ 0       │
```
"""
simulate(m, n::Integer=1) = simulate(Random.default_rng(), m, n)
function simulate(rng::AbstractRNG, m::LPhyloBDP, n::Integer=1)
    p = ProfileSim(m)
    X = mapreduce(x->simulate_profile(rng, m, p), hcat, 1:n) |> permutedims
    DataFrame(X, p.cols)
end

function simulate(rng::AbstractRNG, ms::ModelArray, ::Integer=-1)
    p = ProfileSim(ms.models[1])
    X = mapreduce(m->simulate_profile(rng, m, p), hcat, ms.models) |> permutedims
    DataFrame(X, p.cols)
end

function simulate(rng::AbstractRNG, m::MixtureModel, n::Integer=1)
    p = ProfileSim(m.components[1])
    randmodel = ()->sample(rng, m.components, Weights(m.prior.p))
    X = mapreduce(x->simulate_profile(rng, randmodel(), p), hcat, 1:n) |> permutedims
    DataFrame(X, p.cols)
end

function getleafindex(m)
    Dict(name(n)=>i for (i,n) in enumerate(getleaves(root(m))))    
end

# Should use dispatch on a type which stores those clades...
function getcondition(m, idx)
    o = getroot(m)
    if m.cond == :root
        clades = [name.(getleaves(o[1])), name.(getleaves(o[2]))]
    elseif m.cond == :nowhere
        clades = [name.(getleaves(o))]
    else  # no conditioning
        clades = []
    end
    x->length(clades) == 0 ? true :  
            all([any([x[idx[sp]] > 0 for sp in c]) for c in clades])
end

simulate_profile(m, p) = simulate_profile(Random.default_rng(), m, p)
function simulate_profile(rng::AbstractRNG, m, p::ProfileSim)
    @unpack idx, cond = p
    i = -1
    j = 0
    profile = zeros(Int64, length(idx)+2)
    while i < 0 || !cond(profile)
        full = simwalk!(rng, profile, m, root(m), idx)
        !(cond(profile)) && all(profile .== 0) && (j += 1)
        i += 1
    end
    profile[end-1] = i
    profile[end] = j
    return profile
end

function simwalk!(rng::AbstractRNG, profile, m, n, idx, X=nothing)
    # simulate current edge
    θ  = getθ(m.rates, n)
    X′ = isnothing(X) ?  # root 
        randroot(rng, m.rates.rootprior, θ) : 
        randedge(rng, X, θ, distance(n)) 
    if isleaf(n) 
        profile[idx[name(n)]] = X′
        return X′ 
    end
    next = map(c->simwalk!(rng, profile, m, c, idx, X′), children(n))
    vcat(X′, next...)
end

# only for shifted geometric and geometric priors
randroot(rng::AbstractRNG, prior::Symbol, θ) = 
    rand(rng, Geometric(θ.η)) + (prior == :shifted ? 1 : 0)

function randedge(rng::AbstractRNG, X, θ, t)
    @unpack λ, μ, κ = θ
    (X == 0 && κ == zero(t)) && return 0
    r = κ/λ
    # p ≡ extinction probability of a single lineage
    # q ≡ probability of change conditional on non-extinction
    if λ ≈ μ
        p = q = λ*t/(one(t)+λ*t)
    else
        p = μ*(1.0 - exp((λ-μ)*t))/(μ - λ*exp((λ-μ)*t))
        q = (λ/μ)*p
    end
    #@info "params" X p q r
    X′ = 0
    for i=1:X  # birth-death
        u = rand()
        X′+= u < p ? 0 : rand(rng, Geometric(one(q)-q)) + 1
    end  
    if r > zero(r)  # gain
        X′+= rand(NegativeBinomial(rng, r, one(q)-q))
    end
    return X′
end

