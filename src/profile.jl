# The profile (non-DAG) implementation, should be an abstract matrix
struct Profile{T,I}
    x::Vector{I}
    ℓ::Vector{Vector{T}}
end
Base.length(p::Profile) = length(p.x)
Base.show(io::IO, p::Profile{T,I}) where {T,I} =
    write(io, "Profile{$T,$I}($(p.x), $(sum(p.ℓ[1])))")

struct ProfileMatrix{T,I} <: AbstractMatrix{Profile{T,I}}
    profiles::Vector{Profile{T,I}}
end

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
    ProfileMatrix([Profile(profile[i,:], getparts(profile[i,:], T)) for i=1:N])
end

getparts(x, T) = [fill(T(-Inf), y+1) for y in x]

Base.getindex(P::ProfileMatrix, i) = P.profiles[i]
Base.getindex(P::ProfileMatrix, i, j) = P.profiles[i].x[j]
Base.size(P::ProfileMatrix) = (length(P.profiles), length(P[1]))
nfamilies(P::ProfileMatrix) = size(P)[1]

@inline function cm!(profile::Profile, n, model) where T
    # n is a node from the model
    @unpack x, ℓ = profile
    bound = length(x)
    if isleaf(n)  # leaf case
        return ℓ[id(n)][x[id(n)]] = zero(T)
    end
    kids = children(n)
    kmax = [x[id(k)] for k in kids]
    kcum = cumsum([0 ; kmax])
    keps = [getϵ(c, 1) for c in kid]
    ϵcum = cumprod([1.; keps])
    B = fill(T(-Inf), (bound, kcum[end]+1, length(kmax)))
    A = fill(T(-Inf), (kcum[end]+1, length(kmax)))
    for (i, kid) in enumerate(kids)
        @unpack W = model[id(kid)].data
        cm_inner!(i, A, B, W, parts[id(kid)],
            ϵcum, kcum, kmax[i], log(keps[i]))
    end
    ℓ[id(n)] = A[:,end]
end

function loglikelihood!(p::Profile, model)
    @unpack η = getθ(model.rates, root(model))
    ϵ = log(probify(getϵ(root(model), 2)))
    for n in model.order
        cm!(p, n, model)
    end
    ℓ = ∫rootgeometric(p.ℓ, η, ϵ)
    isfinite(ℓ) ? ℓ : -Inf
end

function loglikelihood!(P::ProfileMatrix{T,I}, model) where {T,I}
    ℓs = zeros(T, nfamilies(P))
    Threads.@threads for i=1:nfamilies(P)
        ℓs[i] = loglikelihood!(p, model)
    end
    ℓ = sum(ℓs) - length(ℓs)*conditionfactor(model)
    isfinite(ℓ) ? ℓ : -Inf
end


# # I would prefer to share the cm routine with the DAG implementation
# # This is the 'classical' implementation, operating on a single (extended) profile NOTE: possible optimizations: (1) matrix operations instead of some loops (not likely to improve speed?)
# @inline function cm!(
#         L::Matrix{T},
#         x::Vector{Int64},
#         n::ModelNode{T}) where T<:Real
#     # @unpack W, ϵ = n.data
#     xmax = maximum(x)
#     if isleaf(n)
#         L[x[id(n)]+1, id(n)] = 0.
#     else
#         kids = children(n)
#         cmax = [x[id(c)] for c in kids]
#         ccum = cumsum([0 ; cmax])
#         ϵcum = cumprod([1.; [getϵ(c, 1) for c in kids]])
#         # XXX possible numerical issues with ϵcum?
#         B = fill(-Inf, (xmax+1, ccum[end]+1, length(cmax)))
#         A = fill(-Inf, (ccum[end]+1, length(cmax)))
#         for i = 1:length(cmax)
#             c  = kids[i]
#             mi = cmax[i]
#             Wc = c.data.W[1:xmax+1, 1:xmax+1]
#             @inbounds B[:, 1, i] = log.(Wc * exp.(L[1:xmax+1, id(c)]))
#             ϵ₁ = log(getϵ(c, 1))
#             for t=1:ccum[i], s=0:mi  # this is 0...M[i-1] & 0...mi
#                 @inbounds B[s+1,t+1,i] = s == mi ?
#                     B[s+1,t,i] + ϵ₁ : logaddexp(B[s+2,t,i], ϵ₁+B[s+1,t,i])
#             end
#             if i == 1
#                 l1me = log(one(ϵ₁) - ϵcum[2])
#                 for n=0:ccum[i+1]  # this is 0 ... M[i]
#                     @inbounds A[n+1,i] = B[n+1,1,i] - n*l1me
#                 end
#             else
#                 # XXX is this loop as efficient as it could? I guess not...
#                 p = probify(ϵcum[i])
#                 for n=0:ccum[i+1], t=0:ccum[i]
#                     s = n-t
#                     (s < 0 || s > mi) && continue
#                     @inbounds lp = binomlogpdf(n, p, s) +
#                         A[t+1,i-1] + B[s+1,t+1,i]
#                     @inbounds A[n+1,i] = logaddexp(A[n+1,i], lp)
#                 end
#                 l1me = log(one(ϵ₁) - ϵcum[i+1])
#                 for n=0:ccum[i+1]  # this is 0 ... M[i]
#                     @inbounds A[n+1,i] -= n*l1me
#                 end
#             end
#         end
#         # @show A[:,end]
#         # if not filling in a matrix, A[:,end] should be the output vector I
#         # guess. The length of this vector would simultaneously specify the
#         # maximum bound for the node
#         for i=0:x[id(n)]
#             @inbounds L[i+1, id(n)] = A[i+1,end]
#         end
#     end
# end
