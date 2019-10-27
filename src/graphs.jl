abstract type GraphGenerator end

struct BarabasiAlbertGenerator <: GraphGenerator
    l::Int
    n::Int
    k::Int
end

LightGraphs.nv(ba::BarabasiAlbertGenerator) = ba.n

param(ba::BarabasiAlbertGenerator) = ba.k

Base.iterate(ba::BarabasiAlbertGenerator) = if ba.l > 0
    g = barabasi_albert(ba.n, ba.k)
    while !is_connected(g)
        barabasi_albert(ba.n, ba.k)
    end
    g, ba.l
else
    nothing
end

Base.iterate(ba::BarabasiAlbertGenerator, i::Int) = if i > 1
    g = barabasi_albert(ba.n, ba.k)
    while !is_connected(g)
        barabasi_albert(ba.n, ba.k)
    end
    g, i - 1
else
    nothing
end

struct ErdosRenyiGenerator <: GraphGenerator
    l::Int
    n::Int
    p::Float64
end

LightGraphs.nv(er::ErdosRenyiGenerator) = er.n

param(er::ErdosRenyiGenerator) = er.p

Base.iterate(er::ErdosRenyiGenerator) = if er.l > 0
    g = erdos_renyi(er.n, er.p)
    while !is_connected(g)
        g = erdos_renyi(er.n, er.p)
    end
    g, er.l
else
    nothing
end

Base.iterate(er::ErdosRenyiGenerator, i::Int) = if i > 1
    g = erdos_renyi(er.n, er.p)
    while !is_connected(g)
        g = erdos_renyi(er.n, er.p)
    end
    g, i - 1
else
    nothing
end

Base.length(er::ErdosRenyiGenerator) = er.l

