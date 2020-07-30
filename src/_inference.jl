# XXX obsolete since Turing has MLE support now

"""
    mle_problem(dag, model)

Obtain an objective function and a function for its gradient
for a given (data, model) pair. The data shoud be stored as
a `CountDAG`.
"""
function mle_problem(dag, model)
    t = model.rates.trans
    function f(x::Vector{T}) where T<:Real
        # rates = model.rates(t(x))
        !(all(isfinite.(x))) && return Inf
        m = model(t(x))
        d = copydag(dag, T)
        -loglikelihood!(d, m)
    end
    return (f=f, ∇f=(G, x) -> G .= ForwardDiff.gradient(f, x))
end

abstract type Problem end

function loglikelihood(p::Problem, θ)
    @unpack model, data = p
    params = model.rates(θ)
    m = PhyloBDP(params, model.order[end], model.bound)
    d = copydag(data, eltype(params))
    loglikelihood!(d, m)
end

trans(p::Problem) = p.model.rates.trans
