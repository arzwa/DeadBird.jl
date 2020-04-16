
# This is the 'classical' implementation, operating on a single (extended) profile NOTE: possible optimizations: (1) matrix operations instead of some loops (not likely to improve speed?)
@inline function cm!(
        L::Matrix{T},
        x::Vector{Int64},
        n::ModelNode{T}) where T<:Real
    # @unpack W, ϵ = n.data
    xmax = maximum(x)
    if isleaf(n)
        L[x[id(n)]+1, id(n)] = 0.
    else
        kids = children(n)
        cmax = [x[id(c)] for c in kids]
        ccum = cumsum([0 ; cmax])
        ϵcum = cumprod([1.; [getϵ(c, 1) for c in kids]])
        # XXX possible numerical issues with ϵcum?
        B = fill(-Inf, (xmax+1, ccum[end]+1, length(cmax)))
        A = fill(-Inf, (ccum[end]+1, length(cmax)))
        for i = 1:length(cmax)
            c  = kids[i]
            mi = cmax[i]
            Wc = c.data.W[1:xmax+1, 1:xmax+1]
            @inbounds B[:, 1, i] = log.(Wc * exp.(L[1:xmax+1, id(c)]))
            ϵ₁ = log(getϵ(c, 1))
            for t=1:ccum[i], s=0:mi  # this is 0...M[i-1] & 0...mi
                @inbounds B[s+1,t+1,i] = s == mi ?
                    B[s+1,t,i] + ϵ₁ : logaddexp(B[s+2,t,i], ϵ₁+B[s+1,t,i])
            end
            if i == 1
                l1me = log(one(ϵ₁) - ϵcum[2])
                for n=0:ccum[i+1]  # this is 0 ... M[i]
                    @inbounds A[n+1,i] = B[n+1,1,i] - n*l1me
                end
            else
                # XXX is this loop as efficient as it could? I guess not...
                p = probify(ϵcum[i])
                for n=0:ccum[i+1], t=0:ccum[i]
                    s = n-t
                    (s < 0 || s > mi) && continue
                    @inbounds lp = binomlogpdf(n, p, s) +
                        A[t+1,i-1] + B[s+1,t+1,i]
                    @inbounds A[n+1,i] = logaddexp(A[n+1,i], lp)
                end
                l1me = log(one(ϵ₁) - ϵcum[i+1])
                for n=0:ccum[i+1]  # this is 0 ... M[i]
                    @inbounds A[n+1,i] -= n*l1me
                end
            end
        end
        # @show A[:,end]
        # if not filling in a matrix, A[:,end] should be the output vector I
        # guess. The length of this vector would simultaneously specify the
        # maximum bound for the node
        for i=0:x[id(n)]
            @inbounds L[i+1, id(n)] = A[i+1,end]
        end
    end
end
