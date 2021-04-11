# Compute likelihood using truncated rate matrices


function QLinearBDP(λ, μ, bound)
    dd = [i == bound-1 ? -i * λ : -i*(λ + μ) for i=0:bound-1]
    dl = [i*μ for i=1:bound-1]
    du = [i*λ for i=0:bound-2]
    Tridiagonal(dl, dd, du)
end
