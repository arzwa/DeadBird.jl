# Benchmarks
To be used with Literate, to generate a markdown report.
Note, apparently, Literate cannot handle begin/end blocks?

```julia
using BenchmarkTools, DeadBird, CSV, DataFrames, Distributions, NewickTree
import DeadBird: loglikelihood!, cm!
```

```
┌ Info: Precompiling DeadBird [a6db34b6-ea8f-496d-a49f-6ef7ed1dec48]
└ @ Base loading.jl:1278
┌ Warning: Package DeadBird does not have ForwardDiff in its dependencies:
│ - If you have DeadBird checked out for development and have
│   added ForwardDiff as a dependency but haven't updated your primary
│   environment's manifest file, try `Pkg.resolve()`.
│ - Otherwise you may need to report an issue with DeadBird
└ Loading ForwardDiff into DeadBird from project dependency, future warnings for DeadBird are suppressed.
WARNING: Method definition loglikelihood(DeadBird.PhyloBDP{T, M, I} where I where M where T, Any) in module DeadBird at /home/arzwa/dev/DeadBird/src/countdag.jl:147 overwritten at /home/arzwa/dev/DeadBird/src/profile.jl:54.
  ** incremental compilation may be fatally broken for this module **


```

The current date/time is

```julia
using Dates
now()
```

```
2020-09-28T09:51:28.979
```

Benchmark on Drosophila

```julia
datadir = "example/drosophila"
df = CSV.read("$datadir/counts-oib.csv")
tr = readnw(readline("$datadir/tree.nw"))
benchmarks = map([1000:1000, 10:10, 2000:100:3000, 1:10]) do idx
    dag, bound = CountDAG(df[idx,:], tr)
    mat, bound = ProfileMatrix(df[idx,:], tr)
    parms = ConstantDLG(λ=1.1, μ=1.2, κ=0.1, η=1/1.5)
    rates = RatesModel(parms, fixed=(:η,:κ))
    model = PhyloBDP(rates, tr, bound)
    bmmat = @benchmark loglikelihood!($(mat), $(model))
    bmdag = @benchmark loglikelihood!($(dag), $(model))
    (idx=idx, dag=bmdag, mat=bmmat)
end
```

```
4-element Array{NamedTuple{(:idx, :dag, :mat),Tuple{StepRange{Int64,Int64},BenchmarkTools.Trial,BenchmarkTools.Trial}},1}:
 (idx = 1000:1:1000, dag = 88.011 μs, mat = 64.622 μs)
 (idx = 10:1:10, dag = 3.214 ms, mat = 3.176 ms)
 (idx = 2000:100:3000, dag = 116.427 μs, mat = 193.089 μs)
 (idx = 1:1:10, dag = 110.074 ms, mat = 82.077 ms)
```

A function to print trials in Literate script

```julia
function print_trials(bmarks)
    b = IOBuffer()
    display.(Ref(TextDisplay(b)), bmarks)
    print(String(take!(b)))
end
```

```
print_trials (generic function with 1 method)
```

For the DAG struct

```julia
print_trials(map(x->x.dag, benchmarks))
```

```
BenchmarkTools.Trial: 
  memory estimate:  89.08 KiB
  allocs estimate:  868
  --------------
  minimum time:     88.011 μs (0.00% GC)
  median time:      127.501 μs (0.00% GC)
  mean time:        135.457 μs (5.75% GC)
  maximum time:     7.488 ms (96.19% GC)
  --------------
  samples:          10000
  evals/sample:     1BenchmarkTools.Trial: 
  memory estimate:  1.32 MiB
  allocs estimate:  2168
  --------------
  minimum time:     3.214 ms (0.00% GC)
  median time:      5.375 ms (0.00% GC)
  mean time:        5.465 ms (1.40% GC)
  maximum time:     21.769 ms (0.00% GC)
  --------------
  samples:          915
  evals/sample:     1BenchmarkTools.Trial: 
  memory estimate:  217.97 KiB
  allocs estimate:  2286
  --------------
  minimum time:     116.427 μs (0.00% GC)
  median time:      199.950 μs (0.00% GC)
  mean time:        225.795 μs (14.56% GC)
  maximum time:     26.914 ms (99.01% GC)
  --------------
  samples:          10000
  evals/sample:     1BenchmarkTools.Trial: 
  memory estimate:  77.90 MiB
  allocs estimate:  37180
  --------------
  minimum time:     110.074 ms (0.00% GC)
  median time:      128.291 ms (0.00% GC)
  mean time:        128.251 ms (2.17% GC)
  maximum time:     144.532 ms (8.78% GC)
  --------------
  samples:          39
  evals/sample:     1
```

For the `ProfileMatrix`

```julia
print_trials(map(x->x.mat, benchmarks))
```

```
BenchmarkTools.Trial: 
  memory estimate:  57.78 KiB
  allocs estimate:  649
  --------------
  minimum time:     64.622 μs (0.00% GC)
  median time:      76.445 μs (0.00% GC)
  mean time:        83.788 μs (6.35% GC)
  maximum time:     4.728 ms (96.45% GC)
  --------------
  samples:          10000
  evals/sample:     1BenchmarkTools.Trial: 
  memory estimate:  1.29 MiB
  allocs estimate:  1946
  --------------
  minimum time:     3.176 ms (0.00% GC)
  median time:      5.262 ms (0.00% GC)
  mean time:        5.010 ms (1.14% GC)
  maximum time:     8.159 ms (0.00% GC)
  --------------
  samples:          998
  evals/sample:     1BenchmarkTools.Trial: 
  memory estimate:  533.86 KiB
  allocs estimate:  6615
  --------------
  minimum time:     193.089 μs (0.00% GC)
  median time:      306.403 μs (0.00% GC)
  mean time:        382.679 μs (22.27% GC)
  maximum time:     39.530 ms (99.08% GC)
  --------------
  samples:          10000
  evals/sample:     1BenchmarkTools.Trial: 
  memory estimate:  77.87 MiB
  allocs estimate:  36956
  --------------
  minimum time:     82.077 ms (0.00% GC)
  median time:      93.529 ms (0.00% GC)
  mean time:        94.062 ms (2.16% GC)
  maximum time:     101.285 ms (4.96% GC)
  --------------
  samples:          54
  evals/sample:     1
```

Comparison of the pruning algorithm on a truncates state space with the CM
algorithm

```julia
#basedir = "example"
#X, s = readdlm("$basedir/9dicots-f01-100.csv", ',', Int, header=true)
#tree = readnw(readline("$basedir/9dicots.nw"))
#r = [0.25, 0.2]
#η = rand(Beta(6,2))
#for bound in [10,25,50,100]
```

   dag, b = CountDAG(X, s, tree)
   rates  = RatesModel(ConstantDLG(λ=r[1], μ=r[2], κ=.0, η=η))
   model1 = PhyloBDP(rates, tree, b)
   ℓ1 = DeadBird.loglikelihood!(dag, model1)
   t1 = @benchmark DeadBird.loglikelihood!($(dag), $(model1))
   @printf "cm: ℓ = %.3f, t = %6.3f, m = %.3f\n" ℓ1 mean(t1.times)/1000 mean(t1.allocs)/1000
   dag_   = DeadBird.nonlineardag(dag, bound)
   rates  = RatesModel(ConstantDLSC(λ=r[1], μ=r[2], μ₁=r[2], η=η, m=bound))
   model2 = PhyloBDP(rates, tree, bound)
   ℓ2 = DeadBird.loglikelihood!(dag_, model2)
   t2 = @benchmark DeadBird.loglikelihood!($(dag_), $(model2))
   m, a, n = mean(t2.times)/1000 mean(t2.allocs)/1000 size(model2.rates.params.Q)[1]
   @printf "tr: ℓ = %.3f, t = %6.3f, m = %.3f, bound = %d\n\n" ℓ2 m a n

```julia
#end
```

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

