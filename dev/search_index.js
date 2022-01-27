var documenterSearchIndex = {"docs":
[{"location":"api/#API","page":"API","title":"API","text":"","category":"section"},{"location":"api/","page":"API","title":"API","text":"Modules = [DeadBird]\nOrder   = [:function, :type]","category":"page"},{"location":"api/#DeadBird.add_internal!-NTuple{5, Any}","page":"API","title":"DeadBird.add_internal!","text":"add_internal!(dag, ndata, parts, x, n)\n\nFor a species tree internal node n, this adds the gene family nodes associated with n to the graph and provides the bound on the number of lineages that survive to the present below n for each gene family.  Note that x is a vector of tuples of DAG nodes that each will be joined into a newly added node.  The resulting nodes are returned.\n\n!!! note: I believe this also works for multifurcating species trees (like the Csuros Miklos algorithm does too)\n\n\n\n\n\n","category":"method"},{"location":"api/#DeadBird.add_leaves!-NTuple{5, Any}","page":"API","title":"DeadBird.add_leaves!","text":"add_leaves!(dag, ndata, parts, x, n)\n\nFor a species tree leaf node n, this adds the vector of (gene) counts x for that species to the graph.  This returns for each gene family the corresponding node that was added to the graph\n\n\n\n\n\n","category":"method"},{"location":"api/#DeadBird.cm!-Union{Tuple{T}, Tuple{CountDAG{T}, Any, Any}} where T","page":"API","title":"DeadBird.cm!","text":"cm!(dag, node, model)\n\nCompute the conditional survival probabilities at n using the Csuros & Miklos (2009) algorithm.  This assumes the model already contains the computed transition probability matrices W and that the partial loglikelihood vectors for the child nodes in the DAG are already computed and available.\n\n\n\n\n\n","category":"method"},{"location":"api/#DeadBird.conditionfactor-Tuple{Any}","page":"API","title":"DeadBird.conditionfactor","text":"conditionfactor(model)\n\nCompute the condition factor for the model for the associated data filtering strategy. \n\n\n\n\n\n","category":"method"},{"location":"api/#DeadBird.discretize-Tuple{Any, Any}","page":"API","title":"DeadBird.discretize","text":"discretize(d, K)\n\nDiscretize a distribution d in K equal probability classes.  Uses the median of each class as representative rate, and rescales the resulting vector x so that mean(x) == mean(d). \n\nnote: Note\nBetter would be to have the mean value of each class as representative I guess, but the median is much more straightforward to obtain given that we have quantile functions available.\n\nExample\n\njulia> discretize(Gamma(10, 0.1), 5)\n5-element Array{Float64,1}:\n 0.6269427439826725\n 0.8195837806573205\n 0.9743503743962694\n 1.1475354999847722\n 1.4315876009789656\n\n\n\n\n\n","category":"method"},{"location":"api/#DeadBird.example_data-Tuple{}","page":"API","title":"DeadBird.example_data","text":"example_data()\n\nGet some example_data.\n\nExample (and benchmark)\n\njulia> x = DeadBird.example_data();\n\njulia> @btime DeadBird.loglikelihood!(x.dag, x.model)\n  36.974 μs (431 allocations: 31.53 KiB)\n-26.30930561857625\n\njulia> @btime DeadBird.loglikelihood!(x.mat, x.model)\n  32.876 μs (420 allocations: 29.91 KiB)\n-26.309305618576246\n\n\n\n\n\n","category":"method"},{"location":"api/#DeadBird.extp-NTuple{4, Any}","page":"API","title":"DeadBird.extp","text":"extp(t, λ, μ, ϵ)\n\nCompute the extinction probability of a single lineage evolving according to a linear BDP for time t with rate λ and μ and with extinction probability of a single lineage at t equal to ϵ. This is ∑ᵢℙ{Xₜ=i|X₀=1}ϵ^i\n\nnote: Note\nTakes ϵ on a [0,1] scale\n\n\n\n\n\n","category":"method"},{"location":"api/#DeadBird.getθ-Union{Tuple{M}, Tuple{M, Any}} where M<:DeadBird.RatesModel","page":"API","title":"DeadBird.getθ","text":"getθ(m<:RatesModel, node)\n\nGet the parameter values from a RatesModel relevant for a particular node in a phylogeny. Should be implemented for each RatesModel where parameters differ across branches.\n\n\n\n\n\n","category":"method"},{"location":"api/#DeadBird.getϕψ-Tuple{Any, Any, Any}","page":"API","title":"DeadBird.getϕψ","text":"getϕψ(t, λ, μ)\n\nReturns ϕ = μ(eʳ - 1)/(λeʳ - μ) where r = t*(λ-μ) and ψ = ϕ*λ/μ, with special cases for λ ≈ μ. These methods should be implemented as to prevent underflow/overflow issues.  Note these quantities are also called p and q (in Csuros & Miklos) or α and β (in Bailey). Note that ϕ = P(Xₜ=0|X₀=1), i.e. the extinction probability for a single gene.\n\n\n\n\n\n","category":"method"},{"location":"api/#DeadBird.getϕψ′-Tuple{Any, Any, Any}","page":"API","title":"DeadBird.getϕψ′","text":"getϕψ′(ϕ, ψ, ϵ)\n\nAdjusted ϕ and ψ for a linear BDP process with extinction probability ϵ after the process.\n\nphi = fracpsi(1-epsilon) + (1 - psi) epsilon1 - psi epsilon\n\npsi = fracpsi(1-epsilon)1 - psi epsilon\n\nSome edge cases are when ϵ is 1 or 0. Other edge cases may be relevant when ψ and or ϕ is 1 or 0.\n\nnote: Note\nWe take ϵ on [0,1] scale.\n\n\n\n\n\n","category":"method"},{"location":"api/#DeadBird.insertwgms-Union{Tuple{T}, Tuple{PhyloBDP{T}, Any}} where T","page":"API","title":"DeadBird.insertwgms","text":"insertwgms(model, wgms::Dict)\n\nInsert a bunch of WGMs in a given PhyloBDP model, will return a new model object. wgms should be a dict with vectors of tuples, keeping for each branch a vector with (t, k) tuples. This version does not modify anything in the template model.\n\nThis is not particularly convenient for use in rjMCMC algorithms, where we want to efficiently add and remove single WGM events...\n\nExample\n\nx = DeadBird.example_data()\nm = PhyloBDP(RatesModel(ConstantDLGWGD(q=ones(9))), x.tr, 5)\ninsertwgms(m, Dict(3=>[(0.1, 2)], 2=>[(0.3, 4)]))\n\n\n\n\n\n","category":"method"},{"location":"api/#DeadBird.loglikelihood!-Union{Tuple{T}, Tuple{CountDAG, PhyloBDP{T, M} where M<:(DeadBird.RatesModel)}} where T","page":"API","title":"DeadBird.loglikelihood!","text":"loglikelihood!(dag::CountDAG, model::PhyloBDP)\n\nCompute the log likelihood on the DAG using the Csuros & Miklos algorithm.\n\n\n\n\n\n","category":"method"},{"location":"api/#DeadBird.marginal_extinctionp","page":"API","title":"DeadBird.marginal_extinctionp","text":"marginal_extinctionp(d, logϵ)\n\nCompute the marginal log probability of having no observed descendants for a branching process starting off with n genes given by a probability distribution d  when the probability that a single gene goes extinct is ϵ. This is:\n\nsum_k=1^infty ϵ^k PX₀ = k\n\nFor many priors a closed form can be obtained by manipulating the sum so that it becomes a geometric series.\n\n\n\n\n\n","category":"function"},{"location":"api/#DeadBird.marginalize","page":"API","title":"DeadBird.marginalize","text":"marginalize(p::ShiftedBetaGeometric, ℓvec, logϵ, imax=100)\n\nThere seems to be no closed form for this, but we can devise a recursion, analogous to the ShiftedBetaGeometric case. This could probably share code, but I'll have it separate for now.\n\n\n\n\n\n","category":"function"},{"location":"api/#DeadBird.marginalize-2","page":"API","title":"DeadBird.marginalize","text":"marginalize(p::ShiftedBetaGeometric, ℓvec, logϵ, imax=100)\n\nThere seems to be no closed form for this, but we can devise a recursion and obtain a near-exact solution efficiently.\n\n\n\n\n\n","category":"function"},{"location":"api/#DeadBird.marginalize-3","page":"API","title":"DeadBird.marginalize","text":"marginalize(prior, ℓvec, logϵ)\n\nCompute the log-likelihood of the data at the root by marginalizing the partial conditional survival likelihoods at the root over the prior on the number of genes at the root. This is the following sum\n\nell = sum_n=1^b elln Big( sum_i=0^infty \n    binomn+ii epsilon^i (1 - epsilon)^n PX_o = n+i Big)\n\nWhere ℓ[n] = P{data|Yₒ=n}, where Yₒ denotes the number of genes at the root that leave observed descendants and Xₒ denotes the total number of genes at the root, for which we specified the prior. b is the bound on the number of surviving lineages, which is determined by the observed data. For many priors, the innner infinite sum can be simplified to a closed form after some algebraic manipulation and relying on the fact that ∑ binom(α + k - 1, k) z^k = (1 - z)^(-k) for |z| < 1.\n\n\n\n\n\n","category":"function"},{"location":"api/#DeadBird.newmodel-Union{Tuple{M}, Tuple{M, Any}} where M<:DeadBird.RatesModel","page":"API","title":"DeadBird.newmodel","text":"newmodel(m::M, θ) where M<:RatesModel\n\nConstruct a new model of type M by taking the parameters of m and parameters defined in the named tuple θ, the latter overriding the former.\n\n\n\n\n\n","category":"method"},{"location":"api/#DeadBird.nonextinctfromrootcondition-Tuple{Any}","page":"API","title":"DeadBird.nonextinctfromrootcondition","text":"nonextinctfromrootcondition(model)\n\nCompute the probability that a family existing at the root of the species tree leaves observed descendants in both clades stemming from the root, i.e. does not go extinct in any of the two clades stemming from the root. This uses the marginalization of the extinction probability over the prior distribution on the number of genes at the root using marginal_extinctionp\n\n\n\n\n\n","category":"method"},{"location":"api/#DeadBird.ppmf-Tuple{NewickTree.Node{I, DeadBird.NodeProbs{T}} where {T, I}, Profile, Any, Any}","page":"API","title":"DeadBird.ppmf","text":"ppmf(node, x, prior, bound)\n\nCompute the posterior pmf for the ancestral state at node node, where x holds the partial likelihoods somewhere, and prior corresponds to the root or transition probability density that acts as a prior on the node state.\n\n\n\n\n\n","category":"method"},{"location":"api/#DeadBird.sample_ancestral-Tuple{DeadBird.AncestralSampler, Any}","page":"API","title":"DeadBird.sample_ancestral","text":"sample_ancestral(spl::AncestralSampler, x)\n\nSample a set of ancestral states using a pre-order traversal over the tree. This assumes the partial likelihoods are available in x.\n\n\n\n\n\n","category":"method"},{"location":"api/#DeadBird.sample_ancestral_node-Tuple{Random.AbstractRNG, Any, Any, Any, Any}","page":"API","title":"DeadBird.sample_ancestral_node","text":"sample_ancestral_node(rng, node, x, prior, bound)\n\nSample ancestral state for node node with prior prior and relevant partial likelihoods computed in x. The prior refers to either a root prior distribution or the transient probability distribution of the process given the parent state.\n\n\n\n\n\n","category":"method"},{"location":"api/#DeadBird.setw!-Union{Tuple{T}, Tuple{AbstractMatrix{T}, Any, Any}} where T","page":"API","title":"DeadBird.setw!","text":"setw!(W, θ, t)\n\nCompute transition probability matrix for the ordinary (not conditional on survival that is) birth-death process. Using the recursive formulation of Csuros & Miklos.\n\n\n\n\n\n","category":"method"},{"location":"api/#DeadBird.simulate","page":"API","title":"DeadBird.simulate","text":"simulate(mfun::Function, data::DataFrame, chain, N)\n\nPerform posterior predictive simulations. mfun should be a function that takes an iterate of the chain and returns a PhyloBDP model,  i.e. mfun(chain[i]) should return a parameterized model. data is the observed data set to which posterior predictive simulations should correspond.\n\nExample\n\njulia> x = DeadBird.example_data();\n\njulia> DeadBird.simulate(y->x.model((λ=y, μ=y)), x.df, ones(10), 100)\nPP simulations (N = 100, n = 10)\n\n\n\n\n\n","category":"function"},{"location":"api/#DeadBird.simulate-2","page":"API","title":"DeadBird.simulate","text":"simulate(m::ModelArray)\nsimulate(m::MixtureModel, n)\nsimulate(m::PhyloBDP, n)\n\nSimulate a set of random profiles from a phylogenetic birth-death model.\n\nExample\n\njulia> x = DeadBird.example_data();\n\njulia> simulate(x.model, 5)\n5×5 DataFrame\n│ Row │ A     │ B     │ C     │ rejected │ extinct │\n│     │ Int64 │ Int64 │ Int64 │ Int64    │ Int64   │\n├─────┼───────┼───────┼───────┼──────────┼─────────┤\n│ 1   │ 1     │ 1     │ 1     │ 0        │ 0       │\n│ 2   │ 1     │ 1     │ 1     │ 0        │ 0       │\n│ 3   │ 2     │ 2     │ 2     │ 0        │ 0       │\n│ 4   │ 0     │ 1     │ 1     │ 1        │ 1       │\n│ 5   │ 1     │ 1     │ 1     │ 0        │ 0       │\n\n\n\n\n\n","category":"function"},{"location":"api/#DeadBird.wgmϵ-Tuple{Any, Any, Any}","page":"API","title":"DeadBird.wgmϵ","text":"wgmϵ(q, k, logϵ)\n\nCompute the log-extinction probability of a single lineage going through a k-plication event, given the extinction probability of a single lineage after the WGM event. Assumes the single parameter WGM retention model (assuming a single gene before the WGM, the number of retained genes after the WGM is a rv X' = 1 + Y where Y is Binomial(k - 1, q)).\n\n\n\n\n\n","category":"method"},{"location":"api/#DeadBird.wstar!-Union{Tuple{T}, Tuple{AbstractMatrix{T}, Any, Any, Any}} where T","page":"API","title":"DeadBird.wstar!","text":"wstar!(W::Matrix, t, θ, ϵ)\n\nCompute the transition probabilities for the conditional survival process recursively (not implemented using recursion though!). Note that the resulting transition matrix is not a stochastic matrix of some Markov chain.\n\n\n\n\n\n","category":"method"},{"location":"api/#DeadBird.AncestralSampler","page":"API","title":"DeadBird.AncestralSampler","text":"AncestralSampler(model, bound)\n\nA wrapper that contains the transition probability matrices for the transient distributions for the PhyloBDP (not conditioned on survival) along each branch.\n\n\n\n\n\n","category":"type"},{"location":"api/#DeadBird.BetaGeometric","page":"API","title":"DeadBird.BetaGeometric","text":"BetaGeometric(η, ζ)\n\n\n\n\n\n","category":"type"},{"location":"api/#DeadBird.ConstantDLG","page":"API","title":"DeadBird.ConstantDLG","text":"ConstantDLG{T}\n\nSimple constant rates duplication-loss and gain model. All nodes of the tree are associated with the same parameters (duplication rate λ, loss rate μ, gain rate κ). \n\n\n\n\n\n","category":"type"},{"location":"api/#DeadBird.ConstantDLGWGM","page":"API","title":"DeadBird.ConstantDLGWGM","text":"ConstantDLGWGM{T}\n\nSimilar to ConstantDLG, but with a field for whole-genome multiplication (WGM) nodes in the phylogeny, which have a single retention rate parameter q each.\n\n\n\n\n\n","category":"type"},{"location":"api/#DeadBird.CountDAG","page":"API","title":"DeadBird.CountDAG","text":"CountDAG(df::DataFrame, tree::Node)\n\nGet a CountDAG from a count matrix, i.e. the directed acyclic graph (DAG) representation of a phylogenetic profile for an (assumed known) species tree. This is a multitree. \n\nExample\n\njulia> x = DeadBird.example_data();\n\njulia> dag = CountDAG(x.df, x.tr)\n(dag = CountDAG({17, 20} directed simple Int64 graph), bound = 7)\n\n\n\n\n\n","category":"type"},{"location":"api/#DeadBird.DLG","page":"API","title":"DeadBird.DLG","text":"DLG{T}\n\nSimple branch-wise rates duplication-loss and gain model.  \n\n\n\n\n\n","category":"type"},{"location":"api/#DeadBird.DLGWGM","page":"API","title":"DeadBird.DLGWGM","text":"DLGWGM{T}\n\nSimilar to DLG, but with WGM nodes, see also ConstantDLGWGM.\n\n\n\n\n\n","category":"type"},{"location":"api/#DeadBird.NodeData","page":"API","title":"DeadBird.NodeData","text":"NodeData{I}\n\nKeeps some relevant information for nodes in the DAG representation of a phylogenetic profile matrix.\n\n\n\n\n\n","category":"type"},{"location":"api/#DeadBird.PPSim","page":"API","title":"DeadBird.PPSim","text":"PPPSim\n\nContainer for posterior predictive simulations, constructor should not be called directly nor exported.\n\n\n\n\n\n","category":"type"},{"location":"api/#DeadBird.PhyloBDP","page":"API","title":"DeadBird.PhyloBDP","text":"PhyloBDP(ratesmodel, tree, bound)\n\nThe phylogenetic birth-death process model as defined by Csuros & Miklos (2009). The bound is exactly defined by the data under consideration.\n\n!!! note: implemented as a <: DiscreteMultivariateDistribution (for     convenience with Turing.jl), however does not support a lot of the     Distributions.jl interface.\n\nExample\n\njulia> x = DeadBird.example_data();\n\njulia> rates = RatesModel(ConstantDLG(λ=0.1, μ=0.1));\n\njulia> dag, bound = CountDAG(x.df, x.tr);\n\njulia> rates = RatesModel(ConstantDLG(λ=0.1, μ=0.1));\n\njulia> PhyloBDP(rates, x.tr, bound)\nPhyloBDP(\n~root\nRatesModel with () fixed\nConstantDLG{Float64}\n  λ: Float64 0.1\n  μ: Float64 0.1\n  κ: Float64 0.0\n  η: Float64 0.66\n)\n\n\n\n\n\n","category":"type"},{"location":"api/#DeadBird.Profile","page":"API","title":"DeadBird.Profile","text":"Profile{T,I}\n\nA phylogenetic profile, i.e. an observation of a discrete random variable associated with the leaves of a phylogenetic tree. This has a field x for the extended profile (which records the bound on the number of lineages that survive below an internal node for internal nodes) and a field for the 'partial likelihoods' ℓ.\n\n\n\n\n\n","category":"type"},{"location":"api/#DeadBird.ProfileMatrix","page":"API","title":"DeadBird.ProfileMatrix","text":"ProfileMatrix(df::DataFrame, tree)\n\nObtain a ProfileMatrix struct for a count dataframe.\n\nExample\n\njulia> x = DeadBird.example_data();\n\njulia> mat, bound = ProfileMatrix(x.df, x.tr)\n(matrix = Profile{Float64,Int64}[2 1 … 0 1; 3 2 … 1 1; 7 3 … 0 4; 7 3 … 3 4], bound = 7)\n\n\n\n\n\n","category":"type"},{"location":"api/#DeadBird.RatesModel","page":"API","title":"DeadBird.RatesModel","text":"RatesModel\n\nAbstract type for diferent rate models for phylogenies (e.g. constant rates across the tree, branch-specific rates, models with WGD nodes, ...).\n\n\n\n\n\n","category":"type"},{"location":"api/#DeadBird.ShiftedBetaGeometric","page":"API","title":"DeadBird.ShiftedBetaGeometric","text":"ShiftedBetaGeometric(η, ζ)\n\nBeta-Geometric compound distribution on the domain [1, 2, ..., ∞).  The pmf is given by\n\np_k = fracmathrmB(alpha + 1 beta + k - 1)mathrmB(alpha beta)\n\nnote: Note\nWe use the alternative parameterization using the mean η = α/(α+β) and offset 'sample size' ζ = α + β - 1, where ζ > 0. That is, we assume α + β > 1.\n\n\n\n\n\n","category":"type"},{"location":"api/#DeadBird.ShiftedGeometric","page":"API","title":"DeadBird.ShiftedGeometric","text":"ShiftedGeometric\n\nGeometric distribution with domain [1, 2, ..., ∞).\n\n\n\n\n\n","category":"type"},{"location":"api/#DeadBird.Transient","page":"API","title":"DeadBird.Transient","text":"Transient\n\nTransient distribution P(X(t)|X(0)=k). This is a simple struct for the sampler.\n\n\n\n\n\n","category":"type"},{"location":"drosophila/#*Drosophila*","page":"Drosophila","title":"Drosophila","text":"","category":"section"},{"location":"drosophila/","page":"Drosophila","title":"Drosophila","text":"Here I illustrate the usage of the DeadBird package for fitting phylogenetic birth-death process models to data using Maximum likelihood and Bayesian inference. We will fit a simple single-rate (turnover rate λ, as in e.g. CAFE) model to the 12 Drosophila species data set.","category":"page"},{"location":"drosophila/","page":"Drosophila","title":"Drosophila","text":"Load the required packages","category":"page"},{"location":"drosophila/","page":"Drosophila","title":"Drosophila","text":"using DeadBird\nusing Distributions, Turing, CSV, DataFrames, NewickTree, Optim\nusing Random; Random.seed!(761);\nnothing #hide","category":"page"},{"location":"drosophila/","page":"Drosophila","title":"Drosophila","text":"Load the data","category":"page"},{"location":"drosophila/","page":"Drosophila","title":"Drosophila","text":"datadir = joinpath(@__DIR__, \"../../example/drosophila\")\ntree = readnw(readline(joinpath(datadir, \"tree.nw\")))\ndata = CSV.read(joinpath(datadir, \"counts-oib.csv\"), DataFrame);\nnothing #hide","category":"page"},{"location":"drosophila/","page":"Drosophila","title":"Drosophila","text":"The data set size and number of taxa are","category":"page"},{"location":"drosophila/","page":"Drosophila","title":"Drosophila","text":"nrow(data), length(getleaves(tree))","category":"page"},{"location":"drosophila/","page":"Drosophila","title":"Drosophila","text":"We'll take a subset of the data for the sake of time.","category":"page"},{"location":"drosophila/","page":"Drosophila","title":"Drosophila","text":"data = data[20:10:10010,:];\nfirst(data, 5)","category":"page"},{"location":"drosophila/","page":"Drosophila","title":"Drosophila","text":"The average number of genes in non-extinct families is","category":"page"},{"location":"drosophila/","page":"Drosophila","title":"Drosophila","text":"m = mean(filter(x->x>0,Matrix(data)))","category":"page"},{"location":"drosophila/","page":"Drosophila","title":"Drosophila","text":"We can use this to parameterize the prior for the number of ancestral lineages","category":"page"},{"location":"drosophila/","page":"Drosophila","title":"Drosophila","text":"η = 1/m\nrootprior = ShiftedGeometric(η)","category":"page"},{"location":"drosophila/","page":"Drosophila","title":"Drosophila","text":"We will use the DAG data structure (most efficient, but admits no family-specific models).","category":"page"},{"location":"drosophila/","page":"Drosophila","title":"Drosophila","text":"dag, bound = CountDAG(data, tree)","category":"page"},{"location":"drosophila/","page":"Drosophila","title":"Drosophila","text":"We will define a Turing model for this simple problem","category":"page"},{"location":"drosophila/","page":"Drosophila","title":"Drosophila","text":"@model singlerate(dag, bound, tree, rootprior) = begin\n    λ ~ Turing.FlatPos(0.)\n    θ = ConstantDLG(λ=λ, μ=λ, κ=zero(λ))\n    dag ~ PhyloBDP(θ, rootprior, tree, bound)\nend","category":"page"},{"location":"drosophila/#Maximum-likelihood-inference","page":"Drosophila","title":"Maximum likelihood inference","text":"","category":"section"},{"location":"drosophila/","page":"Drosophila","title":"Drosophila","text":"First we show how to conduct MLE of a single parameter model for the entire data (i.e. we estimate a genome-wide parameter) using the CountDAG data structure.","category":"page"},{"location":"drosophila/","page":"Drosophila","title":"Drosophila","text":"model = singlerate(dag, bound, tree, rootprior)\n@time mleresult = optimize(model, MLE())","category":"page"},{"location":"drosophila/","page":"Drosophila","title":"Drosophila","text":"For the complete data set of >10000 families, this takes about 10 seconds.","category":"page"},{"location":"drosophila/","page":"Drosophila","title":"Drosophila","text":"It is straightforward to adapt the model definition to allow for different duplication and loss rates, non-zero gain rates (κ) or different root priors.","category":"page"},{"location":"drosophila/","page":"Drosophila","title":"Drosophila","text":"Alternatively we could use the ProfileMatrix, which admits models that deal with variation across families. We can also use this to fit models independently across families. Here we will estimate the MLE of a single turnover rate for 100 families independently.","category":"page"},{"location":"drosophila/","page":"Drosophila","title":"Drosophila","text":"matrix, bound = ProfileMatrix(data, tree)\n\n@model singlerate(mat, bound, tree, rootprior) = begin\n    λ ~ Turing.FlatPos(0.)\n    θ = ConstantDLG(λ=λ, μ=λ, κ=zero(λ))\n    mat ~ PhyloBDP(θ, rootprior, tree, bound)\nend\n\n@time results = map(1:size(matrix, 1)) do i\n    x = matrix[i]\n    model = singlerate(x, x.x[1], tree, rootprior)\n    mleresult = optimize(model, MLE())\n    mleresult.lp, mleresult.values[1]\nend;\nnothing #hide","category":"page"},{"location":"drosophila/","page":"Drosophila","title":"Drosophila","text":"Here we have fitted a single parameter model to each count vector (a phylogenetic profile) independently. Note that the MLE will be zero under this model when the profile consists of ones only. The results of the above look like this","category":"page"},{"location":"drosophila/","page":"Drosophila","title":"Drosophila","text":"first(results, 10)","category":"page"},{"location":"drosophila/","page":"Drosophila","title":"Drosophila","text":"alternatively we can use MAP estimation to regularize the λ estimates, for instance:","category":"page"},{"location":"drosophila/","page":"Drosophila","title":"Drosophila","text":"@model singlerate_ln(mat, bound, tree, rootprior) = begin\n    λ ~ LogNormal(log(0.1), 1)\n    θ = ConstantDLG(λ=λ, μ=λ, κ=zero(λ))\n    mat ~ PhyloBDP(θ, rootprior, tree, bound)\nend\n\n@time results_map = map(1:size(matrix, 1)) do i\n    x = matrix[i]\n    model = singlerate_ln(x, x.x[1], tree, rootprior)\n    mleresult = optimize(model, MAP())\n    mleresult.lp, mleresult.values[1]\nend;\nnothing #hide","category":"page"},{"location":"drosophila/","page":"Drosophila","title":"Drosophila","text":"MAP is also faster, since it is numerically better behaved (the MLE of 0. being on the boundary of parameter space).","category":"page"},{"location":"drosophila/#Bayesian-inference","page":"Drosophila","title":"Bayesian inference","text":"","category":"section"},{"location":"drosophila/","page":"Drosophila","title":"Drosophila","text":"Now we'll perform Bayesian inference using the No-U-turn sampler. Note that we've defined an uninformative flat prior (FlatPos(0.0)), so we expect to find a posterior mean estimate for λ that coincides with the MLE.","category":"page"},{"location":"drosophila/","page":"Drosophila","title":"Drosophila","text":"chain = sample(model, NUTS(), 100)","category":"page"},{"location":"drosophila/","page":"Drosophila","title":"Drosophila","text":"Of course, it would be better to run such a chain for more iterations, e.g. 1000, but for the sake of time I'm only taking a 100 samples here. The 95% uncertainty interval for the turnover rate can be obtained as","category":"page"},{"location":"drosophila/","page":"Drosophila","title":"Drosophila","text":"quantile(chain; q=[0.025, 0.975])","category":"page"},{"location":"drosophila/#Other-models","page":"Drosophila","title":"Other models","text":"","category":"section"},{"location":"drosophila/","page":"Drosophila","title":"Drosophila","text":"It is straightforward to use the Turing.jl model syntax (using the @model macro) in combination with the various rates models and root priors defined in DeadBird (ConstantDLG, DLG, DLGWGM, ShiftedBetaGeometric, ...) to specify complicated models. A not so complicated example would be the following. First we filter the data to only allow for non-extinct families:","category":"page"},{"location":"drosophila/","page":"Drosophila","title":"Drosophila","text":"nonextinct = filter(x->all(Array(x) .> 0), data);\nnothing #hide","category":"page"},{"location":"drosophila/","page":"Drosophila","title":"Drosophila","text":"We will model the excess number of genes, i.e. the number of extra (duplicated) genes per family, instead of the total number of genes.","category":"page"},{"location":"drosophila/","page":"Drosophila","title":"Drosophila","text":"excessgenes = nonextinct .- 1;\nnothing #hide","category":"page"},{"location":"drosophila/","page":"Drosophila","title":"Drosophila","text":"Again we construct a DAG object","category":"page"},{"location":"drosophila/","page":"Drosophila","title":"Drosophila","text":"dag, bound = CountDAG(excessgenes, tree)","category":"page"},{"location":"drosophila/","page":"Drosophila","title":"Drosophila","text":"The model we specify is a linear birth-death and immigration process with immigration (gain) rate equal to the duplication rate, κ = λ, and loss rate μ. This corresponds to a model where genes duplicate at rate λ, (note that a 0 -> 1 transition is also a duplication here since the zero state corresponds to a single copy family), and where duplicated genes get lost at rate μ. We assume λ < μ, in which case there is a geometric stationary distribution with mean 1 - λ/μ for the excess number of genes in a family.","category":"page"},{"location":"drosophila/","page":"Drosophila","title":"Drosophila","text":"bound01(η) = η <= zero(η) ? zero(η) + 1e-16 : η >= one(η) ? one(η) - 1e-16 : η\n\n@model nonextinctmodel(dag, bound, tree) = begin\n    μ ~ Turing.FlatPos(0.)\n    η ~ Beta(1, 1)  # 1 - λ/μ\n    η = bound01(η)\n    λ = μ * (1 - η)\n    rates = ConstantDLG(λ=λ, μ=μ, κ=λ)\n    rootp = Geometric(η)\n    dag ~ PhyloBDP(rates, rootp, tree, bound, cond=:none)\nend","category":"page"},{"location":"drosophila/","page":"Drosophila","title":"Drosophila","text":"and we sample","category":"page"},{"location":"drosophila/","page":"Drosophila","title":"Drosophila","text":"chain = sample(nonextinctmodel(dag, bound, tree), NUTS(), 100)","category":"page"},{"location":"drosophila/","page":"Drosophila","title":"Drosophila","text":"Get the posterior as a dataframe","category":"page"},{"location":"drosophila/","page":"Drosophila","title":"Drosophila","text":"pdf = DataFrame(chain)\nμs = pdf[:, :μ]\nλs = μs .* (1 .- pdf[:,:η])","category":"page"},{"location":"drosophila/","page":"Drosophila","title":"Drosophila","text":"The marginal posterior mean duplication rate (and 95% uncertainty interval) is","category":"page"},{"location":"drosophila/","page":"Drosophila","title":"Drosophila","text":"mean(λs), quantile(λs, [0.025, 0.975])","category":"page"},{"location":"drosophila/","page":"Drosophila","title":"Drosophila","text":"The marginal posterior mean loss rate per duplicated gene is","category":"page"},{"location":"drosophila/","page":"Drosophila","title":"Drosophila","text":"mean(μs), quantile(μs, [0.025, 0.975])","category":"page"},{"location":"drosophila/","page":"Drosophila","title":"Drosophila","text":"","category":"page"},{"location":"drosophila/","page":"Drosophila","title":"Drosophila","text":"This page was generated using Literate.jl.","category":"page"},{"location":"#DeadBird-documentation","page":"Index","title":"DeadBird documentation","text":"","category":"section"},{"location":"#Model-structure","page":"Index","title":"Model structure","text":"","category":"section"},{"location":"","page":"Index","title":"Index","text":"The main object of in this package is the PhyloBDP model, which bundles","category":"page"},{"location":"","page":"Index","title":"Index","text":"a phylogenetic tree\na specification of the duplication, loss and gain rates across the tree\na prior on the number of lineages at the root","category":"page"},{"location":"","page":"Index","title":"Index","text":"In addition, the PhyloBDP model object requires the bound on the number of lineages at the root that leave observed descendants. This bound is determined by the data, and is returned by the functions that read in data in DeadBird.","category":"page"},{"location":"","page":"Index","title":"Index","text":"using DeadBird, NewickTree, DataFrames","category":"page"},{"location":"","page":"Index","title":"Index","text":"First the data side of things:","category":"page"},{"location":"","page":"Index","title":"Index","text":"data = DataFrame(:A=>[1,2], :B=>[0,1], :C=>[3,3])\ntree = readnw(\"((A:1.0,B:1.0):0.5,C:1.5);\")\ndag, bound = CountDAG(data, tree)","category":"page"},{"location":"","page":"Index","title":"Index","text":"Now we specify the model","category":"page"},{"location":"","page":"Index","title":"Index","text":"rates = ConstantDLG(λ=0.5, μ=0.4, κ=0.1)\nprior = ShiftedGeometric(0.9)\nmodel = PhyloBDP(rates, prior, tree, bound)","category":"page"},{"location":"","page":"Index","title":"Index","text":"The model allows likelihood based-inference","category":"page"},{"location":"","page":"Index","title":"Index","text":"using Distributions\nloglikelihood(model, dag)","category":"page"},{"location":"#Data-structures","page":"Index","title":"Data structures","text":"","category":"section"},{"location":"","page":"Index","title":"Index","text":"There are two main data structures to represent the count data.","category":"page"},{"location":"","page":"Index","title":"Index","text":"(1) There is the CountDAG, which efficiently reduces the data to minimize the required computations when all families (rows) share the same model parameters.","category":"page"},{"location":"","page":"Index","title":"Index","text":"dag, bound = CountDAG(data, tree)","category":"page"},{"location":"","page":"Index","title":"Index","text":"(2) There is the ProfileMatrix, which can be used when model parameters are different across families (rows).","category":"page"},{"location":"","page":"Index","title":"Index","text":"mat, bound = ProfileMatrix(data, tree)","category":"page"},{"location":"","page":"Index","title":"Index","text":"Both give identical results","category":"page"},{"location":"","page":"Index","title":"Index","text":"loglikelihood(model, dag) == loglikelihood(model, mat)","category":"page"},{"location":"","page":"Index","title":"Index","text":"","category":"page"},{"location":"","page":"Index","title":"Index","text":"This page was generated using Literate.jl.","category":"page"}]
}
