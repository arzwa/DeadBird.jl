struct PostPredSim{T}
    model    ::T
    posterior::DataFrame
    paramfun ::Function
end

function sim(pp::PostPredSim, N, M;
        sumfun=identity, condition=:root)
    @unpack posterior, model, paramfun = pp
    n, m = size(pp.posterior)
    map(1:N) do i
        x = posterior[rand(1:n),:]
        model′ = paramfun(model, x)
        simout = dlsimbunch(model′, M, minn=2, condition=condition)
        stats = sumfun(simout.profiles)
        (stats=stats, simulation=simout)
    end
end

function leafpmf(df, xmax)
    function f(column)
        xs = counts(column, 0:xmax)
        exceeding = length(filter(x->x>xmax, column))
        [xs ; exceeding]
    end
    Dict(col=>f(df[!,col]) for col in names(df))
end

function pppval(y, ys)
    p = sum(y .> ys)/length(ys)
    p > 0.5 ? one(p) - p : p
end
