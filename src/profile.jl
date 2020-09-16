# The profile (non-DAG) implementation
struct Profile{T,I}
    x::Vector{I}
    ℓ::Vector{Vector{T}}
end
Profile(x, ::Type{T}=Float64) where T = Profile(x, getparts(x, T))
Profile(x, n::Int, ::Type{T}=Float64) where T = Profile(x, getparts(x, n, T))
Base.length(p::Profile) = length(p.x)
Base.show(io::IO, p::Profile{T,I}) where {T,I} =
    write(io, "Profile{$T,$I}($(p.x), $(sum(p.ℓ[1])))")
(p::Profile{T,I})(::Type{V}) where {T,I,V} = Profile(p.x, copyparts(p.ℓ, V))
(p::Profile{T,I})(n, ::Type{V}=T) where {T,I,V} = Profile(p.x, getparts(p.x, n, V))

struct ProfileMatrix{T,I} <: AbstractMatrix{Profile{T,I}}
    profiles::Vector{Profile{T,I}}
end
(p::ProfileMatrix)(::Type{T}) where T = ProfileMatrix([p[i](T) for i=1:nfamilies(p)])
(p::ProfileMatrix{T})(n, ::Type{V}=T) where {T,V} =
    ProfileMatrix([p[i](n+1, V) for i=1:nfamilies(p)])
nonlinearprofile(p, bound) = p(bound)

ProfileMatrix(df, tree) = ProfileMatrix(Matrix(df), names(df), tree)
function ProfileMatrix(matrix::Matrix, names, tree, T=Float64)
    colindex = Dict(string(s)=>i for (i,s) in enumerate(names))
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
    M = ProfileMatrix([Profile(profile[i,:]) for i=1:N])
    (matrix=M, bound=maximum(profile))
end

getparts(x, T) = [fill(T(-Inf), y+1) for y in x]
getparts(x, m, T) = [fill(T(-Inf), m) for y in x]
copyparts(x, T) = [fill(T(-Inf), length(y)) for y in x]

Base.getindex(P::ProfileMatrix, i) = P.profiles[i]
Base.getindex(P::ProfileMatrix, i, j) = P.profiles[i].x[j]
Base.size(P::ProfileMatrix) = (length(P.profiles), length(P[1]))
nfamilies(P::ProfileMatrix) = size(P)[1]

Distributions.logpdf(m::PhyloBDP{T}, x::ProfileMatrix) where T = loglikelihood!(x(T), m)
Distributions.logpdf(m::PhyloBDP{T}, x::Profile) where T = loglikelihood!(x(T), m)
Distributions.logpdf(M::ModelArray{<:PhyloBDP{T}}, x::ProfileMatrix) where T = 
    loglikelihood!(x(T), M)
Distributions.loglikelihood(m::PhyloBDP, x) = logpdf(m, x)
Distributions.loglikelihood(M::ModelArray, x) = logpdf(M, x)

# This is shared over both linear/nonlinear models
function loglikelihood!(P::ProfileMatrix{T,I}, model) where {T,I}
    ℓs = zeros(T, nfamilies(P))
    Threads.@threads for i=1:nfamilies(P)
        ℓs[i] = loglikelihood!(P[i], model, false)
    end
    ℓ = sum(ℓs) - length(ℓs)*conditionfactor(model)
    isfinite(ℓ) ? ℓ : -Inf
end

function loglikelihood!(P::ProfileMatrix{T,I}, M::ModelArray) where {T,I}
    ℓs = zeros(T, nfamilies(P))
    Threads.@threads for i=1:nfamilies(P)
        ℓs[i] = loglikelihood!(P[i], M.models[i], false) - 
            conditionfactor(M.models[i])
    end
    ℓ = sum(ℓs)
    isfinite(ℓ) ? ℓ : -Inf
end
