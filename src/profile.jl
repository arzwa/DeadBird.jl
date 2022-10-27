# The profile (non-DAG) implementation
"""
    Profile{T,I}

A phylogenetic profile, i.e. an observation of a discrete random variable
associated with the leaves of a phylogenetic tree. This has a field `x` for the
extended profile (which records the bound on the number of lineages that
survive below an internal node for internal nodes) and a field for the 'partial
likelihoods' `ℓ`.
"""
struct Profile{T,I}
    x::Vector{I}
    ℓ::Vector{Vector{T}}
end
Profile(x, ::Type{T}=Float64) where T = Profile(x, getparts(x, T))
Profile(x, n::Int, ::Type{T}=Float64) where T = Profile(x, getparts(x, n, T))

Base.length(p::Profile) = length(p.x)

function Base.show(io::IO, p::Profile{T,I}) where {T,I}
    write(io, "Profile{$T,$I}($(p.x), $(sum(p.ℓ[1])))")
end

(p::Profile{T,I})(::Type{V}) where {T,I,V} = Profile(p.x, copyparts(p.ℓ, V))
(p::Profile{T,I})(n, ::Type{V}=T) where {T,I,V} = Profile(p.x, getparts(p.x, n, V))

"""
    ProfileMatrix(df::DataFrame, tree)

Obtain a ProfileMatrix struct for a count dataframe.

# Example
```julia-repl
julia> x = DeadBird.example_data();

julia> mat, bound = ProfileMatrix(x.df, x.tr)
(matrix = Profile{Float64,Int64}[2 1 … 0 1; 3 2 … 1 1; 7 3 … 0 4; 7 3 … 3 4], bound = 7)
```
"""
struct ProfileMatrix{T,I} <: AbstractMatrix{Profile{T,I}}
    profiles::Vector{Profile{T,I}}
end

(p::ProfileMatrix)(::Type{T}) where T = ProfileMatrix([p[i](T) for i=1:size(p, 1)])
function (p::ProfileMatrix{T})(n, ::Type{V}=T) where {T,V}
    ProfileMatrix([p[i](n+1, V) for i=1:size(p, 1)])
end

ProfileMatrix(df, tree) = ProfileMatrix(Matrix(df), names(df), tree)

function ProfileMatrix(matrix, names, tree, ::Type{T}=Float64) where T
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

getparts(x, ::Type{T}) where T = [fill(T(-Inf), y+1) for y in x]
getparts(x, m, ::Type{T}) where T = [fill(T(-Inf), m) for y in x]
copyparts(x, ::Type{T}) where T = [fill(T(-Inf), length(y)) for y in x]

Base.getindex(P::ProfileMatrix, i) = P.profiles[i]
Base.getindex(P::ProfileMatrix, i, j) = P.profiles[i].x[j]
Base.size(P::ProfileMatrix) = (length(P.profiles), length(P[1]))
Base.iterate(P::ProfileMatrix) = iterate(P.profiles)
Base.iterate(P::ProfileMatrix, i) = iterate(P.profiles, i)

Distributions.logpdf(m::PhyloBDP{T}, x::ProfileMatrix) where T = loglikelihood!(x(T), m)
Distributions.logpdf(m::PhyloBDP{T}, x::Profile) where T = loglikelihood!(x(T), m)
Distributions.logpdf(M::ModelArray{<:PhyloBDP{T}}, x::ProfileMatrix) where T = 
    loglikelihood!(x(T), M)

# This is shared over both linear/nonlinear models
function loglikelihood!(P::ProfileMatrix{T,I}, model) where {T,I}
    ℓs = zeros(T, size(P, 1))
    Threads.@threads for i=1:size(P, 1)
        ℓs[i] = loglikelihood!(P[i], model, false)
    end
    ℓ = sum(ℓs) - length(ℓs)*conditionfactor(model)
    isfinite(ℓ) ? ℓ : -Inf
end

function loglikelihood!(P::ProfileMatrix{T,I}, M::ModelArray) where {T,I}
    ℓs = zeros(T, size(P, 1))
    Threads.@threads for i=1:size(P, 1)
        ℓs[i] = loglikelihood!(P[i], M.models[i], false) - 
            conditionfactor(M.models[i])
    end
    ℓ = sum(ℓs)
    isfinite(ℓ) ? ℓ : -Inf
end
