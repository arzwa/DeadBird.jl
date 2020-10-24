"""
    PPPSim

Container for posterior predictive simulations, constructor should not be
called directly nor exported.
"""
struct PPSim{T,V}
    mfun::Function
    data::T
    sims::V
    N::Int
    n::Int
end

Base.show(io::IO, x::PPSim) = write(io, "PP simulations (N = $(x.N), n = $(x.n))")

function qtable(x::PPSim; sp=Symbol[], xs=1:5)
    sp = isempty(sp) ? collect(keys(x.data)) : sp
    for n=xs
        for (i,k) in enumerate(sp)
            q1 = quantile(x.sims[k][n,:], 0.025)
            q2 = quantile(x.sims[k][n,:], 0.975)
            @printf "%.3f (%.3f, %.3f)" x.data[k][n] q1 q2
            i == length(sp) ? nothing : print(" | ")
        end
        print("\n")
    end
end

"""
    simulate(mfun::Function, data::DataFrame, chain, N)

Perform posterior predictive simulations. `mfun` should be a function
that takes an iterate of the `chain` and returns a `PhyloBDP` model, 
i.e. `mfun(chain[i])` should return a parameterized model. `data` is
the observed data set to which posterior predictive simulations should
correspond.

# Example
```
julia> x = DeadBird.example_data();

julia> DeadBird.simulate(y->x.model((λ=y, μ=y)), x.df, ones(10), 100)
PP simulations (N = 100, n = 10)
```
"""
function simulate(mfun::Function, data::DataFrame, chain, N=nrow(data))    
    simfun = i->simulate(mfun(chain[i]), N) |> leafpmf 
    Ypp = tmap(simfun, 1:length(chain))
    PPSim(mfun, leafpmf(data), combine_pmfs(Ypp), N, length(chain))
end

function simulate_ma(mfun::Function, data::DataFrame, chain)    
    simfun = i->simulate(mfun(chain[i])) |> leafpmf 
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
    Qs[:,3] .= abs.(Qs[:,3] .- Qs[:,1])
    Qs
end

function getmin(sims::PPSim)
    xs = map(x->minimum(x[x .!= 0.]), values(sims.sims))
    xs = vcat(xs, map(x->minimum(x[x .!= 0.]), values(sims.data)))
    minimum(xs)
end

function getsteps(X)
    X = mapslices(x->repeat(x, inner=2), X, dims=1)[1:end-1,:] 
    x = X[2:end,1]
    y = X[1:end-1,2:end]
    (x=x, y=y)
end

function pppvalue(x::PPSim, sp::Symbol, k) 
    p = ecdf(x.sims[sp][k,:])(x.data[sp][k])
    p > 0.5 ? 1 - p : p
end

@recipe function f(::Type{Val{:stepplot}}, plt::AbstractPlot)
    y = plotattributes[:y]
    x = plotattributes[:x]
    a = length(y)
    rib = !isnothing(plotattributes[:ribbon])
    if rib
        l, u = plotattributes[:fillrange]
        X = [x y l u]
    else
        X = [x y]
    end
    x_, y_ = getsteps(X)
    seriestype   := :path
    seriescolor --> :black
    fillalpha   --> 0.1
    grid        --> false
    legend      --> false
    vert = haskey(plotattributes, :vert) ? 
        plotattributes[:vert] : true
    if vert
        x := x_
        y := y_[:,1]
        if rib
            fillrange := (y_[:,2], y_[:,3])
        end
    else
        for i=1:2:length(xs)-1
            @series begin
                x := x_[i:i+1]
                y := y_[i:i+1,1]
                if rib 
                    fillrange := (y_[i:i+1,2], y_[i:i+1,3])
                end
            end
        end
    end
end
@shorthands stepplot

@recipe function f(pps::PPSim)
    @unpack data, sims = pps
    xguide --> "\$n\$"
    yguide --> "\$\\log_{10}p\$"
    guidefont --> 8
    legend --> false
    grid   --> false
    layout --> length(data) 
    xscale --> :log10
    ϵ = 1/2pps.N
    ylims --> (log10(ϵ), 0)
    
    for (i, (k,v)) in enumerate(data)
        sp = join(split(string(k), "_"), " ")
        Qs = quantiles(sims[k])
        x, y = getsteps([1:size(Qs)[1] Qs])
        v[v .== 0.] .= ϵ 
        v = log10.(v)
        c = [j > size(Qs)[1] || !(Qs[j,1] - Qs[j,2] < v[j] < Qs[j,1] + Qs[j,3]) ? 
             :white : :black for j=1:length(v)]

        @series begin
            subplot := i
            title := sp
            titlefontfamily := :italic
            titlefont := 8
            titlelocation --> :left
            seriestype := :path
            seriescolor --> :black
            fillalpha   --> 0.2
            ribbon --> (y[:,2], y[:,3])
            x, y[:,1]
        end

        @series begin
            subplot := i
            title := sp
            titlefontfamily := :italic
            titlefont := 8
            titlelocation --> :left
            markershape --> :circle
            markersize  --> 3
            seriestype := :scatter
            seriescolor --> c
            x = collect(1:length(v)) .+ 0.5
            x, v
        end
    end
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

