# The profile (non-DAG) implementation, should be an abstract matrix
struct Profile{T,I}
    x::Vector{I}
    ℓ::Vector{Vector{T}}
end
Base.length(p::Profile) = length(p.x)
Base.show(io::IO, p::Profile{T,I}) where {T,I} =
    write(io, "Profile{$T,$I}($(p.x), $(sum(p.ℓ[1])))")
(p::Profile{T,I})(::Type{V}) where {T,I,V} = Profile(p.x, getparts(p.x, V))

struct ProfileMatrix{T,I} <: AbstractMatrix{Profile{T,I}}
    profiles::Vector{Profile{T,I}}
end
(p::ProfileMatrix)(::Type{T}) where T =
    ProfileMatrix([p[i](T) for i=1:nfamilies(p)])

ProfileMatrix(df, tree) = ProfileMatrix(Matrix(df), names(df), tree)
function ProfileMatrix(matrix::Matrix, names, tree, T=Float64)
    colindex = Dict(s=>i for (i,s) in enumerate(names))
    N = size(matrix)[1]
    nodes = postwalk(tree)
    profile = zeros(Int, N, length(nodes))
    for n in nodes
        if isleaf(n)
            profile[:,id(n)] = matrix[:,colindex[name(n)]]
        else
            for c in children(n)
                profile[:,id(n)] .+= profile[:,id(c)]
            end
        end
    end
    M = ProfileMatrix([Profile(profile[i,:], getparts(profile[i,:], T)) for i=1:N])
    (matrix=M, bound=maximum(profile)+1)
end

getparts(x, T) = [fill(T(-Inf), y+1) for y in x]

Base.getindex(P::ProfileMatrix, i) = P.profiles[i]
Base.getindex(P::ProfileMatrix, i, j) = P.profiles[i].x[j]
Base.size(P::ProfileMatrix) = (length(P.profiles), length(P[1]))
nfamilies(P::ProfileMatrix) = size(P)[1]

@inline function cm!(profile::Profile{T}, n, model) where T
    # n is a node from the model
    @unpack x, ℓ = profile
    bound = length(x)
    if isleaf(n)  # leaf case
        return ℓ[id(n)][x[id(n)]+1] = zero(T)
    end
    kids = children(n)
    kmax = [x[id(k)] for k in kids]
    kcum = cumsum([0 ; kmax])
    keps = [getϵ(c, 1) for c in kids]
    ϵcum = cumprod([1.; keps])
    B = fill(T(-Inf), (bound, kcum[end]+1, length(kmax)))
    A = fill(T(-Inf), (kcum[end]+1, length(kmax)))
    for (i, kid) in enumerate(kids)
        @unpack W = model[id(kid)].data
        cm_inner!(i, A, B, W, ℓ[id(kid)],
            ϵcum, kcum, kmax[i], log(keps[i]))
    end
    ℓ[id(n)] = A[:,end]
end

function loglikelihood!(p::Profile, model, condition=true)
    @unpack η = getθ(model.rates, root(model))
    ϵ = log(probify(getϵ(root(model), 2)))
    for n in model.order
        cm!(p, n, model)
    end
    ℓ = ∫rootgeometric(p.ℓ[1], η, ϵ)
    if condition
        ℓ -= conditionfactor(model)
    end
    isfinite(ℓ) ? ℓ : -Inf
end

function loglikelihood!(P::ProfileMatrix{T,I}, model) where {T,I}
    ℓs = zeros(T, nfamilies(P))
    Threads.@threads for i=1:nfamilies(P)
        ℓs[i] = loglikelihood!(P[i], model, false)
    end
    ℓ = sum(ℓs) - length(ℓs)*conditionfactor(model)
    isfinite(ℓ) ? ℓ : -Inf
end

Distributions.logpdf(m::PhyloBDP{T}, x::Union{ProfileMatrix,Profile}) where T =
    loglikelihood!(x(T), m)
