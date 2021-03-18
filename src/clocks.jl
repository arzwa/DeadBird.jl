# Molecular clock priors

# I should have a multivariate version of this as well
"""
    BrownianLogClock

Geometric brownian motion prior for log rates along a phylogeny, sensu Yang &
Rannala (2007). Should also work for non-binary (polytomous) trees.

!!! note 
    This applies a drift correction as in Thorne & Kishino and Yang and
    Rannala. It is therefore *not* simply a Brownian motion on the log rates
    (which would entail an increasing expected rate on the original scale).
    Under this model, the expected rate is `exp(μ)` along each branch.
"""
struct BrownianLogClock{T,N} <: ContinuousMultivariateDistribution 
    μ::T
    σ::T
    preorder::Vector{N}
    BrownianLogClock(μ::T, σ::T, tree::N) where {T,N} = new{T,N}(μ, σ, prewalk(tree))
end

BrownianLogClock(μ, σ, tree) = BrownianLogClock(promote(μ, σ)..., tree)

Base.length(d::BrownianLogClock) = length(d.preorder) - 1
Bijectors.bijector(d::BrownianLogClock) = Identity{1}()

# get distances to midpoints of branches above and below a node `n`
distance_to_midpoints(n) = distance(n)/2, distance.(children(n)) ./ 2

# get the MvNormal distribution of the rates at the midpoints of the daughter
# branches given the rate at the midpoint of the parent branch
function branch_dist(ya, ta, tc, σsq)
    η = [ya - σsq*(ta + tc[i]) for i=1:length(tc)]
    Σ = fill(ta, length(tc), length(tc))
    for i=1:length(tc)
        Σ[i,i] = ta + tc[i]
    end
    return MvNormal(η, 2σsq * Σ)
end

function Base.rand(rng::AbstractRNG, d::BrownianLogClock)
    @unpack μ, σ, preorder = d
    x = Vector{Float64}(undef, length(d))
    σsq = (σ*σ)/2
    for n in preorder
        isleaf(n) && continue
        ta, tc = distance_to_midpoints(n) 
        if isroot(n)
            for (i,c) in enumerate(children(n))
                x[id(c)-1] = rand(rng, Normal(d.μ - tc[i] * σsq, σ*√(tc[i])))
            end
        else
            ys = rand(rng, branch_dist(x[id(n)-1], ta, tc, σsq))
            for (i,c) in enumerate(children(n))
                x[id(c)-1] = ys[i]
            end
        end
    end
    return x
end

function Distributions.logpdf(d::BrownianLogClock, x::AbstractVector)
    @unpack μ, σ, preorder = d
    logp = 0.
    σsq = (σ*σ)/2
    for n in preorder
        isleaf(n) && continue
        ta, tc = distance_to_midpoints(n)
        if isroot(n) 
            for (i,c) in enumerate(children(n))
                logp += logpdf(Normal(μ - tc[i] * σsq, σ*√(tc[i])), x[id(c)-1])
            end
        else
            ys  = [x[id(c)-1] for c in children(n)]
            mvn = branch_dist(x[id(n)-1], ta, tc, σsq)
            logp += logpdf(mvn, ys)
        end
    end
    return logp
end

"""
    MvBrownianLogClock

A multivariate geometric brownian motion model with drift correction, specified
for the midpoints of each branch in the tree. This generalize Yang & Rannala's
prior to the multivariate case (i.e. multiple potentially correlated 'traits'
or parameters for each branch).
"""
struct MvBrownianLogClock{T,V,N} <: ContinuousMatrixDistribution
    μ::Vector{T}
    Σ::Matrix{V}
    preorder::Vector{N}
    MvBrownianLogClock(μ::Vector{T}, Σ::Matrix{V}, tree::N) where {T,V,N} =
        new{T,V,N}(μ, Σ, prewalk(tree))
end

Base.size(d::MvBrownianLogClock) = (length(d.preorder)-1, size(d.Σ, 1))
Bijectors.bijector(d::MvBrownianLogClock) = Identity{2}()

function branch_dist(ya, ta, tc, m, Σ)
    nc = length(tc)
    ys = repeat(ya, nc)
    Σa = Σ*ta
    ΣΣ = reduce(vcat, [reduce(hcat, [Σa for i=1:nc]) for i=1:nc])
    idx = 1:m:(m*nc)
    for (i, j) in enumerate(idx)
        ys[j:(j+m-1)] .-= (Σ[i,i]*(ta + tc[i]) / 2)
        ΣΣ[j:(j+m-1),j:(j+m-1)] += Σ*tc[i]
    end
    return idx, MvNormal(ys, ΣΣ)
end

function Base.rand(rng::Random.AbstractRNG, d::MvBrownianLogClock)
    @unpack μ, Σ, preorder = d
    l, m = size(d)
    Y = Matrix{Float64}(undef, l, m)
    for n in preorder
        isleaf(n) && continue
        ta, tc = distance_to_midpoints(n)
        if isroot(n)
            for (i, c) in enumerate(children(n))
                ya = μ - (diag(Σ) .* tc[i] ./ 2)
                Y[id(c)-1,:] = rand(rng, MvNormal(ya, Σ*tc[i]))
            end
        else
            idx, mvn = branch_dist(Y[id(n)-1,:], ta, tc, m, Σ)
            ys = rand(rng, mvn)
            for (i, (j, c)) in enumerate(zip(idx, children(n)))
                Y[id(c)-1,:] = ys[j:(j+m-1)]
            end
        end
    end
    return Y
end

function Distributions.logpdf(d::MvBrownianLogClock, Y::AbstractMatrix{T}) where T<:Real
    @unpack μ, Σ, preorder = d
    l, m = size(d)
    logp = 0.
    for n in preorder
        isleaf(n) && continue
        ta, tc = distance_to_midpoints(n)
        if isroot(n)
            for (i, c) in enumerate(children(n))
                ya = μ - (diag(Σ) .* tc[i] ./ 2)
                logp += logpdf(MvNormal(ya, Σ*tc[i]), Y[id(c)-1,:])
            end
        else
            idx, mvn = branch_dist(Y[id(n)-1,:], ta, tc, m, Σ)
            ys = reduce(vcat, [Y[id(c)-1,:] for c in children(n)])
            logp += logpdf(mvn, ys)
        end
    end
    return logp
end
