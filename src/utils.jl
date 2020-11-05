# utilities for OrthoFinder data processing
function getcountsof(df, tree)
    species = sort(name.(getleaves(tree)))
    fun = x->ismissing(x) ? 0 : length(split(x, ", "))
    ddf = select(df, species .=> ByRow(x->fun(x)))  
    rename!(ddf, species)
    (df=hcat(df[!,Not(species)], ddf), cols=species)
end

function oibfilter(df, tree)
    clade1 = name.(getleaves(tree[1]))
    clade2 = name.(getleaves(tree[2]))
    filter(x->sum(x[clade1]) > 0 && sum(x[clade2]) > 0, df)
end

function nonextinctfilter(df, tree) 
    species = sort(name.(getleaves(tree)))
    (df=filter(x->all(Array(x[species]) .> 0), df), cols=species)
end

# two typical data sets we use are obtained by the following
function getextra(df, tree) 
    df, cols = getcountsof(df, tree)
    df, cols = nonextinctfilter(df, tree)
    df[cols] = df[cols] .-= 1
    (df=df, cols=cols)
end

function getoib(df, tree) 
    df, cols = getcountsof(df, tree)
    (df=oibfilter(df, tree), cols=cols)
end

# Get the orthogroups (i.e. not counts) for the `oib` filtered data set This is
# often the largest subset of the data we work with.
function getoib_orthogroups(df, tree, key=:HOG)
    ddf, cols = getoib(df, tree)
    out = semijoin(df, ddf, on=key)
    out[!, Symbol.([key; cols])]
end

function getne_orthogroups(df, tree, key=:HOG)
    ddf, cols = getcountsof(df, tree)
    ddf, cols = nonextinctfilter(ddf, tree)
    out = semijoin(df, ddf, on=key)
    out[!, Symbol.([key; cols])]
end

"""
    discretize(d, K)

Discretize a distribution `d` in `K` equal probability classes.  Uses the
median of each class as representative rate, and rescales the resulting vector
`x` so that `mean(x) == mean(d)`. 

!!! note
    Better would be to have the mean value of each class as representative
    I guess, but the median is much more striaghtforward to obtain given
    that we have quantile functions available.

# Example

```julia-repl
julia> discretize(Gamma(10, 0.1), 5)
5-element Array{Float64,1}:
 0.6269427439826725
 0.8195837806573205
 0.9743503743962694
 1.1475354999847722
 1.4315876009789656
```
"""
function discretize(d, K)
    qstart = 1.0/2K
    qend = 1. - 1.0/2K
    xs = quantile.(d, qstart:(1/K):qend)
    xs *= mean(d)*K/sum(xs)  # rescale by factor mean(d)/mean(xs)
end


