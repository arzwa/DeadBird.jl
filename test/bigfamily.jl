# The big family problem is really bugging me. 
# For families with large gene counts, ForwardDiff gives NaN or 0.0
# gradients...
# I have been unable to track down the source of this issue.

tx = 4
df = CSV.read("../example/drosophila/$(tx)taxa.csv")
tr = readnw(readline("../example/drosophila/$(tx)taxa.nw"))
dag, bound = CountDAG(df[1:100,:], tr)
rates = RatesModel(ConstantDLG(λ=1e2, μ=1e2, κ=0., η=1/1.5), fixed=(:η,:κ))
model = PhyloBDP(rates, tr, bound)

using ForwardDiff

# MNWE
mat, bound = ProfileMatrix(df[1:1,:], tr)
x = [1.703358267, 1.54850752]
g = ForwardDiff.gradient(x->logpdf(model(x), mat), x)

# I guess the problem is with extremely tiny probabilities that we should
# ignore I guess... leading to entries in the B matrix where the highest
# probabilities are of (log) order like
#   Dual{ForwardDiff.Tag{var"#173#174",Float64}}(-54.5447,126.702,-72.6846)
# whereas the lowest are 
#   Dual{ForwardDiff.Tag{var"#173#174",Float64}}(-723.359,-Inf,Inf)
# but these should actually be 
#   Dual{ForwardDiff.Tag{var"#173#174",Float64}}(-Inf,0,0) 
# I guess...


for i=1:100
    x = repeat(randn(1), 2); x[1] *= 1.1
    g = ForwardDiff.gradient(x->logpdf(model(x), dag), x)
    @info "∇ℓ" x g
    sleep(1)
end

using Optim, Turing
@model mle(model, data) = begin
    r ~ MvLogNormal(zeros(2), ones(2))
    data ~ model((λ=r[1], μ=r[2]))
end

optimize(mle(model, dag), MLE())
