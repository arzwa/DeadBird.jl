# inserts a (WGMk,post-WGMk) node pair at time `t` above `node`
function _insertwgm!(node::ModelNode{T,I}, n, k, t) where {T,I}
    # initially: u -l→ v 
    # the goal:  u -(l-t)→ w -0→ x -t→ v
    # where w is the WGM node and x is the post-WGM node
    u = parent(node)
    delete!(u, node)
    l = distance(node)
    m = getbound(node)
    w = Node(I(n), NodeProbs("", k, l-t, m, T), u)
    x = Node(I(n+1), NodeProbs("", 1, 0., m, T), w)
    v = Node(id(node), NodeProbs(name(node), getk(node), t, m, T), x)
    return [v, x, w] 
end

"""
    insertwgms(model, wgms::Dict)

Insert a bunch of WGMs in a given PhyloBDP model, will return a new model
object. `wgms` should be a dict with vectors of tuples, keeping for each branch
a vector with (t, k) tuples. This version does not modify anything in the
template model.

This is not particularly convenient for use in rjMCMC algorithms, where we want
to efficiently add and remove single WGM events...

# Example

```julia
x = DeadBird.example_data()
m = PhyloBDP(RatesModel(ConstantDLGWGD(q=ones(9))), x.tr, 5)
insertwgms(m, Dict(3=>[(0.1, 2)], 2=>[(0.3, 4)]))
```
"""
function insertwgms(model::PhyloBDP{T}, wgms...) where T
    order = eltype(model.order)[]
    nodes = typeof(model.nodes)()
    rates = deepcopy(model.rates)
    n = length(model)
    m = model.bound
    wgms = collect_and_order(wgms)
    # x is the node to be copied, y is the parent of copy to created
    function walk(x, y)
        ys = [Node(id(x), NodeProbs(x, m, T), y)]
        if haskey(wgms, id(x))
            # iterate over WGMs on the relevant branch starting from the farthest
            for (t, k, q) in wgms[id(x)]
                xs = _insertwgm!(ys[1], n+1, k, t)
                # note that v replaces the relevant daughter node
                rates.q[id(last(xs))] = q
                popfirst!(ys)
                ys = [xs; ys]
                n += 2
            end
        end
        for c in children(x) walk(c, first(ys)) end
        push!(order, ys...)
        for y in ys; nodes[id(y)] = y; end
    end
    walk(getroot(model), nothing)
    model = PhyloBDP(rates, model.rootp, nodes, order, m, model.cond)
    setmodel!(model)  # assume the model should be initialized
    return model
end

function collect_and_order(pairs)
    d = Dict()
    for p in pairs
        !haskey(d, first(p)) ? 
            d[first(p)] = [last(p)] : 
            push!(d[first(p)], last(p))
        sort!(d[first(p)], rev=true)
    end
    return d
end

