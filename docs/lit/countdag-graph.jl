
using Plots, StatsPlots, CSV, DataFrames, Parameters

dir = "/home/arzwa/research/gene-family-evolution/data/oryzinae"
data = CSV.read("$dir/oryza-max10-6taxa-oib.csv", DataFrame)
tree = readnw(readline(joinpath(dir, "oryzinae.6taxa.nw")))

function height(n)
    isleaf(n) ? 0 : maximum(height.(children(n))) + 1
end

function generate_tikz_graph(dag, tree)
    f(i, n) = "$i / \"$(n.snode), $(n.bound)\""
    nodes = Dict{Int,Vector{String}}()
    heights = Dict(id(n)=>height(n) for n in prewalk(tree))
    function walk(graph, i)
        xs = [walk(graph, x) for x in graph.fadjlist[i]]
        ns = f(i, dag.ndata[i])
        #sn = heights[dag.ndata[i].snode]
        sn = dag.ndata[i].snode
        haskey(nodes, sn) ? push!(nodes[sn], ns) : (nodes[sn] = [ns])
        return isempty(xs) ? ns : "$ns -> {$(join(xs, ", "))}"
    end
    paths = [walk(dag.graph, i) for i in dag.levels[end]]
    #levels = ["{ [same layer] $(join(map(i->f(i, dag.ndata[i]), x), ", "))}, " for x in reverse(dag.levels)] 
    #levels = ["{ [same layer] $(join(nodes[k], ", "))}," for k in sort(collect(values(heights)), rev=true)]
    levels = ["{ [same layer] $(join(nodes[id(k)], ", "))}," for k in prewalk(tree)]
    s = "\\usetikzlibrary{graphdrawing,graphs}\n\\usegdlibrary{layered}\n"
    s *= "\\tikz [rounded corners] \\graph [layered layout] {"
    s *= "\n\t" * join(levels, "\n\t")
    s *= "\n\t" * join(paths, ";\n\t")
    s *= "\n};"
    println(s)
end

idx = sample(5000:5200, 10, replace=false)
dag, bound = CountDAG(data[idx,:], tree)
ntree = deepcopy(tree)
for n in postwalk(ntree)
    n.data.name = string(id(n))
end

o = name.(getleaves(tree))
p1 = plot(ntree, internal=true, fontfamily="computer modern", xlim=(0,0.8))
X = permutedims(Matrix(data[idx,o]))
p2 = heatmap(X, color=:binary, xticks=false,
             yticks=false, colorbar=false, framestyle=:none)
vline!(p2, 0.5:10.5, color=:white, legend=false)
hline!(p2, 0.5:6.5, color=:white, legend=false)
for i=1:size(X,1), j=1:size(X,2)
    annotate!(p2, j, i, text("$(X[i,j])", 10, "helvetica bold", :white))
end
plot(p1, p2, size=(400,180), layout=grid(1,2,widths=[0.3,0.7]))
savefig("docs/img/dagtree.pdf")
