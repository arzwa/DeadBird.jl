
**Warning:** This is rapidly evolving research software, and still in a largely
experimental part of its life cycle. The methods should be reliable (see the
tests), but the API may change. Contributions welcome, feel free to reach out.

# DeadBird

`DeadBird.jl` is a julia package for modeling comparative genomic count data
using **phylogenetic birth-death processes**, most commonly gene families.
`DeadBird.jl` uses the (exact) algorithm of Csuros & Miklos (2009) for
computing the conditional survival likelihoods. 

Some things `DeadBird.jl` currently allows to do:

- Flexibly specify models of evolution (e.g. with branch-specific rates using
  molecular clock priors, models of rate heterogeneity across families,
  different prior distributions on the number of lineages at the root, ...).
- Perform Bayesian *inference* and maximum likelihood estimation (using
  automatic differentiation and the Turing library for probabilistic
  programming) of duplication, loss and gain rates for these complex models
  along a known phylogeny.
- Simulate data under these possibly complicated models, and assess model fit
  using posterior predictive simulations.
- Statistically test for whole-genome multiplications along branches of the
  species tree (sensu Rabier et al. 2014, Zwaenepoel & Van de Peer 2019).

`DeadBird` is developed by Arthur Zwaenepoel (member of the Van de Peer group
at VIB-UGent center for plant systems biology). If you use `DeadBird` please
cite the following article (which describes a previous version on which
this package is based)

```
Zwaenepoel, A., and Y. Van de Peer. 
"Model-based detection of whole-genome duplications in a phylogeny." 
Molecular biology and evolution (2020).
```

and consider citing

```
Csűrös, Miklós, and István Miklós. 
"Streamlining and large ancestral genomes in Archaea inferred with a 
phylogenetic birth-and-death model." 
Molecular biology and evolution 26.9 (2009): 2087-2095.
```
