
struct PPSim{T,V}
    mfun::Function
    data::T
    sims::V
    N::Int
    n::Int
end

Base.show(io::IO, x::PPSim) = write(io, "PP simulations (N = $(x.N), n = $(x.n))")

function simulate(mfun::Function, data::DataFrame, chain, N=nrow(data))    
    simfun = i->simulate_profile(mfun(chain[i]), N) |> leafpmf 
    Ypp = tmap(simfun, 1:length(chain))
    PPSim(mfun, leafpmf(data), combine_pmfs(Ypp), N, length(chain))
end

function simulate_ma(mfun::Function, data::DataFrame, chain)    
    simfun = i->simulate_profile(mfun(chain[i])) |> leafpmf 
    Ypp = tmap(simfun, 1:length(chain))
    PPSim(mfun, leafpmf(data), combine_pmfs(Ypp), nrow(data), length(chain))
end

function leafpmf(df) 
    Dict(k=>_proportions(v) for (k, v) in eachcol(df, true))
end

_proportions(x) = proportions(x, 0:maximum(x))

function combine_pmfs(xs::Vector{<:Dict})
    ns = sort(collect(keys(xs[1])))
    Dict(n=>hcatpad(map(x->x[n], xs)) for n in ns)
end

function hcatpad(xs::Vector{Vector{T}}) where T
    maxlen = mapreduce(x->length(x), max, xs)
    newxs  = map(xs) do x
        newx = vcat(x, zeros(T, maxlen - length(x)))
    end 
    hcat(newxs...)
end

function quantiles(sims::Dict; trans=log10, qs=[0.025, 0.975])
    Dict(k=>quantiles(X, trans=trans, qs=qs) for (k,X) in sims)
end

function quantiles(sims::Matrix; trans=log10, qs=[0.025, 0.975])
    ϵ = minimum(sims[sims .!= 0]) / 10
    Yk = trans.(sims .+ ϵ)
    Qs = mapslices(x->vcat(mean(x), quantile(x, [0.025, 0.975])), Yk, dims=2)
    Qs[:,2] .= abs.(Qs[:,2] .- Qs[:,1])
    Qs[:,3] .= Qs[:,3] .- Qs[:,1]
    Qs
end

function getmin(sims::PPSim)
    xs = map(x->minimum(x[x .!= 0.]), values(sims.sims))
    xs = vcat(xs, map(x->minimum(x[x .!= 0.]), values(sims.data)))
    minimum(xs)
end

@userplot PPPlot2
@recipe function f(x::PPPlot2)
    @unpack sims, data, N, n = x.args[1]
    grid --> false
    legend --> false
    xguide --> "\$n\$"
    yguide --> "\$\\epsilon\$"
    layout --> length(data)
    xscale --> :log10
    for (i,(k,v)) in enumerate(data)
        @series begin
            seriestype := :scatter
            seriescolor --> :black
            msim = mapslices(mean, sims[k], dims=2)
            x = (v .- msim[1:length(v)]) ./ v
            subplot := i
            x
        end
        @series begin
            subplot := i
            seriestype := :hline
            seriescolor --> :black
            linestyle --> :dash
            [0.]
        end
    end
end

@userplot StepPlot
@recipe function f(x::StepPlot)
    X = x.args[1]
    vert = length(x.args) == 2 ? x.args[end] : true
    a = size(X)[1]
    X = mapslices(x->repeat(x, inner=2), X, dims=1)[1:end-1,:] 
    xs = repeat(1:a, inner=2)[2:end] .+ 0.
    xs[2:end] .-= 0.5
    seriestype --> :path
    seriescolor --> :black
    fillalpha --> 0.2
    grid --> false
    legend --> false
    if size(X)[2] > 2
        ribbon --> (X[:,2], X[:,3])
    end
    if vert
        xs, X[:,1]
    else
        for i=1:2:length(xs)-1
            @series begin
                xs[i:i+1], X[i:i+1,1]
            end
        end
    end
end

@recipe function f(pps::PPSim)
    @unpack data, sims = pps
    xguide --> "\$n\$"
    yguide --> "\$\\log_{10}p\$"
    guidefont --> 8
    legend --> false
    grid   --> false
    layout --> length(data) 
    yscale --> :log10
    xscale --> :log10
    ϵ = getmin(pps) / 10
    ylims  --> (5ϵ, 0)
    
    for (i, (k,v)) in enumerate(data)
        @series begin
            seriestype := :path
            seriescolor --> :black
            sims[k][:,1] .+ ϵ
        end

        @series begin
            markershape --> :circle
            markersize  --> 3
            seriestype := :scatter
            seriescolor --> :white
            subplot := i
            title := k
            titlefontfamily := :italic
            titlefont := 8
            #yscale := :log10
            v .+ ϵ
        end
    end
end
