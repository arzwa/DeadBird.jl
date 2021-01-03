
# Maximum likelihood and Bayesian inference for the 12 Drosophila data set

Here I illustrate the usage of the DeadBird package for fitting phylogenetic
birth-death process models to data using Maximum likelihood and Bayesian
inference. We will fit a simple single-rate (turnover rate λ, as in e.g.
CAFE) model to the 12 *Drosophila* species data set.

Load the required packages

```@example drosophila
using DeadBird, Distributions, Turing, CSV, DataFrames, NewickTree, Optim
using Random; Random.seed!(671);
nothing #hide
```

Load the data

```@example drosophila
datadir = joinpath(@__DIR__, "../../example/drosophila")
tree = readnw(readline(joinpath(datadir, "tree.nw")))
data = CSV.read(joinpath(datadir, "counts-oib.csv"), DataFrame);
nothing #hide
```

The data set size and number of taxa are

```@example drosophila
nrow(data), length(getleaves(tree))
```

We'll take a subset of the data for the sake of time.

```@example drosophila
data = data[20:10:10010,:];
first(data, 5)
```

The average number of genes in non-extinct families is

```@example drosophila
m = mean(filter(x->x>0,Matrix(data)))
```

We can use this to parameterize the prior for the number of ancestral
lineages

```@example drosophila
η = 1/m
rootprior = ShiftedGeometric(η)
```

We will use the DAG data structure (most efficient).

```@example drosophila
dag, bound = CountDAG(data, tree)
```

We will define a Turing model for this simple problem

```@example drosophila
@model singlerate(dag, bound, tree, rootprior) = begin
    λ ~ Turing.FlatPos(0.)
    θ = ConstantDLG(λ=λ, μ=λ, κ=zero(λ))
    dag ~ PhyloBDP(θ, rootprior, tree, bound)
end
```

## Maximum likelihood inference

```@example drosophila
model = singlerate(dag, bound, tree, rootprior)
@time mleresult = optimize(model, MLE())
```

For the complete data set, this takes a bot 10 seconds.

It is straightforward to adapt the model definition to allow for different
duplication and loss rates, non-zero gain rates (`κ`) or different root
priors.

Now we'll perform Bayesian inference using the No-U-turn sampler. Note that
we've defined an uninformative flat prior, so we expect to find a posterior
mean estimate for `λ` that coincides with the MLE.

```@example drosophila
chain = sample(model, NUTS(), 100)
```

Of course, it would be better to run such a chain for more iterations, e.g.
1000, but for the sake of time I'm only taking a 100 samples here. The 95%
uncertainty interval for the turnover rate can be obtained as

```@example drosophila
quantile(chain; q=[0.025, 0.975])
```

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

