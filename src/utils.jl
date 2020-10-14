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
