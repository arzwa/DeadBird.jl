
# DeadBird documentation

## Model structure

The main object of in this package is the `PhyloBDP` model, which bundles
- a phylogenetic tree
- a specification of the duplication, loss and gain rates across the tree
- a prior on the number of lineages at the root

In addition, the `PhyloBDP` model object requires the bound on the number
of lineages at the root that leave observed descendants. This bound is
determined by the data, and is returned by the functions that read in
data in `DeadBird`.

```@example index
using DeadBird, NewickTree, DataFrames
```

First the data side of things:

```@example index
data = DataFrame(:A=>[1,2], :B=>[0,1], :C=>[3,3])
tree = readnw("((A:1.0,B:1.0):0.5,C:1.5);")
dag, bound = CountDAG(data, tree)
```

Now we specify the model

```@example index
rates = ConstantDLG(λ=0.5, μ=0.4, κ=0.1)
prior = ShiftedGeometric(0.9)
model = PhyloBDP(rates, prior, tree, bound)
```

The model allows likelihood based-inference

```@example index
using Distributions
loglikelihood(model, dag)
```

## Data structures
There are two main data structures to represent the count data.

(1) There is the `CountDAG`, which efficiently reduces the data to minimize
the required computations when all families (rows) share the same model
parameters.

```@example index
dag, bound = CountDAG(data, tree)
```

(2) There is the `ProfileMatrix`, which can be used when model parameters are
different across families (rows).

```@example index
mat, bound = ProfileMatrix(data, tree)
```

Both give identical results

```@example index
loglikelihood(model, dag) == loglikelihood(model, mat)
```

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

