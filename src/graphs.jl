abstract type GraphGenerator end

Base.length(g::GraphGenerator) = g.count

LightGraphs.nv(g::GraphGenerator) = g.nodes

params(g::GraphGenerator) = tuple()

Base.iterate(g::GraphGenerator) = if length(g) > 0
    graph = generate(g)
    while !is_connected(graph)
        graph = generate(g)
    end
    graph, length(g)
else
    nothing
end

Base.iterate(g::GraphGenerator, i::Int) = if i > 1
    graph = generate(g)
    while !is_connected(graph)
        graph = generate(g)
    end
    graph, i - 1
else
    nothing
end

struct CycleGraphGenerator <: GraphGenerator
    count::Int
    nodes::Int
end
CycleGraphGenerator(; count::Int=1, nodes::Int=1, kwargs...) = CycleGraphGenerator(count, nodes)

generate(c::CycleGraphGenerator) = cycle_graph(c.nodes)

struct WheelGraphGenerator <: GraphGenerator
    count::Int
    nodes::Int
end
WheelGraphGenerator(; count::Int=1, nodes::Int=1, kwargs...) = WheelGraphGenerator(count, nodes)

generate(w::WheelGraphGenerator) = wheel_graph(w.nodes)

struct StarGraphGenerator <: GraphGenerator
    count::Int
    nodes::Int
end
StarGraphGenerator(; count::Int=1, nodes::Int=1, kwargs...) = StarGraphGenerator(count, nodes)

generate(s::StarGraphGenerator) = star_graph(s.nodes)

struct GridGraphGenerator <: GraphGenerator
    count::Int
    dims::Vector{Int}
    periodic::Bool
end
GridGraphGenerator(; count::Int=1, dims::Vector{Int}=[1], periodic=true, kwargs...) = GridGraphGenerator(count, dims, periodic)

LightGraphs.nv(g::GridGraphGenerator) = prod(g.dims)

generate(g::GridGraphGenerator) = grid(g.dims; periodic=g.periodic)

struct BarabasiAlbertGenerator <: GraphGenerator
    count::Int
    nodes::Int
    k::Int
end
BarabasiAlbertGenerator(; count::Int=1, nodes::Int=1, k::Int=1, kwargs...) = BarabasiAlbertGenerator(count, nodes, k)

params(ba::BarabasiAlbertGenerator) = (; k=ba.k)

generate(ba::BarabasiAlbertGenerator) = barabasi_albert(ba.nodes, ba.k)

struct ErdosRenyiGenerator <: GraphGenerator
    count::Int
    nodes::Int
    p::Float64
end
ErdosRenyiGenerator(; count::Int=1, nodes::Int=1, p::Float64=0.5, kwargs...) = ErdosRenyiGenerator(count, nodes, p)

params(er::ErdosRenyiGenerator) = (; p=er.p)

generate(er::ErdosRenyiGenerator) = erdos_renyi(er.nodes, er.p)
