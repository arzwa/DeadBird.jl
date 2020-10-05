# Arthur Zwaenepoel (2020)
# Simulate profiles directly for Linear BDPs
# I think this is much more to the point (but does not simulate trees). Probably
# should be the `rand` function. 
# For posterior predictive simulations, this better be fast!
function simulate_profile(m::ModelArray)
    # Assumes a random profile should be drawn for each model in the `ModelArray`
    idx, f, cols = prepare_sim(m[1])
    res = mapreduce(i->simulate_profile(m[i], idx, f), hcat, 1:length(m)) |> permutedims
    DataFrame(res, Symbol.(cols)) 
end

function simulate_profile(m::MixtureModel, n::Integer) 
    idx, f, cols = prepare_sim(m.components[1])
    randmodel = ()->sample(m.components, Weights(m.prior.p))
    res = mapreduce(i->simulate_profile(randmodel(), idx, f), hcat, 1:n) |> permutedims
    DataFrame(res, Symbol.(cols))
end

function simulate_profile(m::PhyloBDP, n::Integer)
    idx, f, cols = prepare_sim(m)
    res = mapreduce(i->simulate_profile(m, idx, f), hcat, 1:n) |> permutedims 
    DataFrame(res, Symbol.(cols))
end

function prepare_sim(model)
    idx = getleafindex(model)
    f = getcondition(model, idx)
    ks = first.(sort(collect(idx), by=x->last(x)))
    cols = vcat(ks..., ["rejected", "extinct"])
    return (idx=idx, f=f, cols=cols)
end

function getleafindex(m)
    Dict(name(n)=>i for (i,n) in enumerate(getleaves(root(m))))    
end

function simulate_profile(m, idx=getleafindex(m), f=getcondition(m, idx))
    i = -1
    j = 0
    profile = zeros(Int64, length(idx)+2)
    while i < 0 || !f(profile)
        full = simwalk!(profile, m, root(m), idx)
        !(f(profile)) && all(profile .== 0) && (j += 1)
        i += 1
    end
    profile[end-1] = i
    profile[end] = j
    return profile
end

function simwalk!(profile, m, n, idx, X=nothing)
    # simulate current edge
    θ  = getθ(m.rates, n)
    X′ = isnothing(X) ?  # root 
        randroot(m.rates.rootprior, θ) : 
        randedge(X, θ, distance(n)) 
    if isleaf(n) 
        profile[idx[name(n)]] = X′
        return X′ 
    end
    next = map(c->simwalk!(profile, m, c, idx, X′), children(n))
    vcat(X′, next...)
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

# only for shifted geometric and geometric priors
randroot(prior::Symbol, θ) = rand(Geometric(θ.η)) + (prior == :shifted ? 1 : 0)

function randedge(X, θ, t)
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
        X′+= u < p ? 0 : rand(Geometric(one(q)-q)) + 1
    end  
    if r > zero(r)  # gain
        X′+= rand(NegativeBinomial(r, one(q)-q))
    end
    return X′
end

