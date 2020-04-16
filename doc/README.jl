
# ## Maximum-likelihood estimation
using BirdDad, Optim, NewickTree, DelimitedFiles

# Read in the data and tree
X, s = readdlm("example/9dicots-f01-1000.csv", ',', Int, header=true)
tree = readnw(readline("example/9dicots.nw"))

# Construct the data object
dag, bound = CountDAG(X, s, tree)

# Construct the model
rates = RatesModel(
    ConstantDLG(λ=1.0, μ=1.2, κ=0.0, η=1/mean(X)),
    fixed=(:η, :κ))
model = PhyloBDP(rates, tree, bound)

f, ∇f = mle_problem(dag, model)
@time out = optimize(f, ∇f, randn(2), BFGS())
t = transform(model.rates.trans, out.minimizer)

# Note that this is an order of magnitude faster than the `R` implementation in WGDgc (where I measured a run time ≈35s for the same data and initial conditions)

# ## Side note:
# The Hessian of the negative loglikelihood evluated at the MLE is equal to the observed Fisher information $I(\hat{\theta})$. The estimated standard errors for the ML estimates $\mathrm{SE}(\hat{\theta}) = 1/\sqrt{I(\hat{\theta})}$ can be obtained as follows
using ForwardDiff, LinearAlgebra

function ff(x::Vector{T}) where T
    rates = model.rates((λ=x[1], μ=x[2]))
    m = PhyloBDP(rates, tree, bound)
    d = copydag(dag, T)
    -loglikelihood!(d, m)
end
1 ./ diag(.√(ForwardDiff.hessian(ff, [t.λ, t.μ])))

# Note that, sadly, we cannot use `ForwardDiff.hessian(f, out.minimizer)` to obtain this, since that would calculate the hessian in transformed space.
