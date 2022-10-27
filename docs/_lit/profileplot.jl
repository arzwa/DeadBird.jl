using DeadBird, CSV, DataFrames, Plots, StatsBase, NewickTree

# Load the data
datadir = joinpath("example/drosophila")
tree = readnw(readline(joinpath(datadir, "tree.nw")))
data = CSV.read(joinpath(datadir, "counts-oib.csv"), DataFrame);

# The data set size and number of taxa are
nrow(data), length(getleaves(tree))

species = Dict("dper"=>"D. persimilis",
               "dpse"=>"D. pseudoobscura",
               "dsec"=>"D. sechellia",
               "dsim"=>"D. simulans",
               "dmel"=>"D. melanogaster",
               "dere"=>"D. erecta",
               "dyak"=>"D. yakuba",
               "dana"=>"D. ananassae",
               "dwil"=>"D. willistoni",
               "dvir"=>"D. virilis",
               "dmoj"=>"D. mojavensis",
               "dgri"=>"D. grimshawi")

data = filter(x->length(unique(Array(x)))!=1, data)

idx = sample(1:nrow(data), 15)
X = Matrix(data[idx,name.(getleaves(tree))]) |> permutedims
t = deepcopy(tree)
for (i,n) in enumerate(postwalk(t))
    n.data.name = @sprintf "%3d" i
end
p1 = plot(t, linecolor=:lightgray,
          internal=true, fontsize=6, linewidth=10, pad=1,
          fontfamily="helvetica", xlim=(-0.05,0.55))
p2 = heatmap(X, color=:binary, colorbar=false, framestyle=:none)
for i=1:size(X,1), j=1:size(X,2)
    annotate!(p2, j, i, text("$(X[i,j])", 10, "helvetica bold", :white))
end
plot(p1, p2, layout=grid(1,2,widths=[0.4,0.6]), size=(600,300))

savefig("docs/img/profile-example.pdf")
