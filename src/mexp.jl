# From:
# https://github.com/sdewaele/GenericLinearAlgebra.jl/commit/53133b5a5f210cb72de73f7cee71a9f0ca2cd362
# This matrix exponential works with AD, (I hope correctly)
# Julia matrix exponential

using LinearAlgebra
using LinearAlgebra:checksquare,copytri!
using GenericLinearAlgebra

## Destructive matrix exponential using algorithm from Higham, 2008,
## "Functions of Matrices: Theory and Computation", SIAM
function _exp!(A::StridedMatrix{T}) where T<:Number
  n = checksquare(A)
  # TODO
  if ishermitian(A)
      return copytri!(parent(exp(Hermitian(A))), 'U', true)
  end
  # TODO Write generic balancing
  # ilo, ihi, scale = LAPACK.gebal!('B', A)    # modifies A
  nA   = opnorm(A, 1)
  Inn    = Matrix{T}(I, n, n)
  ## For sufficiently small nA, use lower order Padé-Approximations
  if (nA <= 2.1)
      if nA > 0.95
          C = T[17643225600.,8821612800.,2075673600.,302702400.,
                   30270240.,   2162160.,    110880.,     3960.,
                         90.,         1.]
      elseif nA > 0.25
          C = T[17297280.,8648640.,1995840.,277200.,
                   25200.,   1512.,     56.,     1.]
      elseif nA > 0.015
          C = T[30240.,15120.,3360.,
                  420.,   30.,   1.]
      else
          C = T[120.,60.,12.,1.]
      end
      A2 = A * A
      P  = copy(Inn)
      U  = C[2] * P
      V  = C[1] * P
      for k in 1:(div(size(C, 1), 2) - 1)
          k2 = 2 * k
          P *= A2
          U += C[k2 + 2] * P
          V += C[k2 + 1] * P
      end
      U = A * U
      X = V + U
      ldiv!(lu!(V-U), X) # LAPACK.gesv!(V-U, X)
  else
      s  = log2(nA/5.4)               # power of 2 later reversed by squaring
      if s > 0
          si = ceil(Int,s)
          A /= 2^si # convert(T,2^si) # convert ==> error: no method matching convert(::Type{BigFloat}, ::Int64)
      end
      CC = T[64764752532480000.,32382376266240000.,7771770303897600.,
              1187353796428800.,  129060195264000.,  10559470521600.,
                  670442572800.,      33522128640.,      1323241920.,
                      40840800.,           960960.,           16380.,
                           182.,                1.]
      A2 = A * A
      A4 = A2 * A2
      A6 = A2 * A4
      U  = A * (A6 * (CC[14].*A6 .+ CC[12].*A4 .+ CC[10].*A2) .+
                CC[8].*A6 .+ CC[6].*A4 .+ CC[4].*A2 .+ CC[2].*Inn)
      V  = A6 * (CC[13].*A6 .+ CC[11].*A4 .+ CC[9].*A2) .+
                 CC[7].*A6 .+ CC[5].*A4 .+ CC[3].*A2 .+ CC[1].*Inn

      X = V + U
      ldiv!(lu!(V-U), X) # LAPACK.gesv!(V-U, X)

      if s > 0            # squaring to reverse dividing by power of 2
          for t=1:si; X *= X end
      end
  end

  # TODO Re-enable this code once generic balancing is in place
  # # Undo the balancing
  # for j = ilo:ihi
  #     scj = scale[j]
  #     for i = 1:n
  #         X[j,i] *= scj
  #     end
  #     for i = 1:n
  #         X[i,j] /= scj
  #     end
  # end

  # if ilo > 1       # apply lower permutations in reverse order
  #     for j in (ilo-1):-1:1; rcswap!(j, Int(scale[j]), X) end
  # end
  # if ihi < n       # apply upper permutations in forward order
  #     for j in (ihi+1):n;    rcswap!(j, Int(scale[j]), X) end
  # end
  X
end

_exp(A::StridedMatrix{<:Number}) = _exp!(copy(A))

LinearAlgebra.exp!(A::StridedMatrix) = _exp!(A)
LinearAlgebra.exp(A::StridedMatrix) = _exp(A)
