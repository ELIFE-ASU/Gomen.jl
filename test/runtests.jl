using Gomen
using Base.MathConstants, Random, Test
using StaticArrays
using LightGraphs, LightGraphs.SimpleGraphs

@testset "Game" begin
    @testset "Construction" begin
        @test_throws DomainError Game(-1, 1)
        @test_throws DomainError Game(1, 1)
        @test_throws DomainError Game(0, 0)
        @test_throws DomainError Game(0, 2)

        @test Game(0.5, 1).payoffs == @SArray [1.0 0.5; 1.0 0.0]
        @test Game(-0.3, 1.3).payoffs == @SArray [1.0 -0.3; 1.3 0.0]
    end

    @testset "Play" begin
        let g = Game(-0.3, 1.3)
            @test play(g, 1, 1) == 1.0
            @test play(g, 1, 2) == -0.3
            @test play(g, 2, 1) == 1.3
            @test play(g, 2, 2) == 0.0
        end
    end
end

@testset "Rule" begin
    @testset "Sigmoid" begin
        let points = [-1.0, -0.5, 0.0, 0.5, 1.0], p = sqrt(e), q = e^0.25
            @test_throws DomainError Sigmoid(-1.0)
            let rule = Sigmoid()
                expect = [1/(1 + e), 1/(1 + p), 0.5, p/(1 + p), e/(1 + e)]
                @test map(dp -> apply(rule, dp), points) ≈ expect
            end
            let rule = Sigmoid(0.5)
                expect = [1/(1 + p), 1/(1 + q), 0.5, q/(1 + q), p/(1 + p)]
                @test map(dp -> apply(rule, dp), points) ≈ expect
            end
        end
    end

    @testset "Heaviside" begin
        let points = [-1.0, -0.5, 0.0, 0.5, 1.0]
            @test_throws DomainError Heaviside(-1.0)
            let rule = Heaviside()
                @test map(dp -> apply(rule, dp), points) ≈ [0.0, 0.0, 0.5, 1.0, 1.0]
            end
            let rule = Heaviside(0.75)
                @test map(dp -> apply(rule, dp), points) ≈ [0.0, 0.5, 0.5, 0.5, 1.0]
            end
        end
    end
end

@testset "Scheme" begin
    @testset "Single Agent" begin
        let N = 1000
            σ(p) = sqrt(N*p*(1-p))
            cond(p) = N*p - 3σ(p), N*p + 3σ(p)

            println("Random Seed: ", Random.GLOBAL_RNG.seed)
            Random.seed!(Random.GLOBAL_RNG.seed)

            let cf = CounterFactual()
                a, b = cond(0.5)
                @test a ≤ sum(Int(decide(cf, 1, 0.0) == 2) for _ in 1:N) ≤ b
                @test a ≤ sum(Int(decide(cf, 2, 0.0) == 1) for _ in 1:N) ≤ b

                a, b = cond(e / (1 + e))
                @test a ≤ sum(Int(decide(cf, 1, 1.0) == 2) for _ in 1:N) ≤ b
                @test a ≤ sum(Int(decide(cf, 2, 1.0) == 1) for _ in 1:N) ≤ b

                a, b = cond(1 / (1 + e))
                @test a ≤ sum(Int(decide(cf, 1, -1.0) == 2) for _ in 1:N) ≤ b
                @test a ≤ sum(Int(decide(cf, 2, -1.0) == 1) for _ in 1:N) ≤ b
            end

            let cf = CounterFactual(Sigmoid(0.5))
                a, b = cond(0.5)
                @test a ≤ sum(Int(decide(cf, 1, 0.0) == 2) for _ in 1:N) ≤ b
                @test a ≤ sum(Int(decide(cf, 2, 0.0) == 1) for _ in 1:N) ≤ b

                a, b = cond(sqrt(e) / (1 + sqrt(e)))
                @test a ≤ sum(Int(decide(cf, 1, 1.0) == 2) for _ in 1:N) ≤ b
                @test a ≤ sum(Int(decide(cf, 2, 1.0) == 1) for _ in 1:N) ≤ b

                a, b = cond(1 / (1 + sqrt(e)))
                @test a ≤ sum(Int(decide(cf, 1, -1.0) == 2) for _ in 1:N) ≤ b
                @test a ≤ sum(Int(decide(cf, 2, -1.0) == 1) for _ in 1:N) ≤ b
            end

            let cf = CounterFactual(Heaviside())
                a, b = cond(0.5)
                @test a ≤ sum(Int(decide(cf, 1, 0.0) == 2) for _ in 1:N) ≤ b
                @test a ≤ sum(Int(decide(cf, 2, 0.0) == 1) for _ in 1:N) ≤ b

                @test a ≤ sum(Int(decide(cf, 1, 1e-4) == 2) for _ in 1:N) ≤ b
                @test a ≤ sum(Int(decide(cf, 2, 1e-4) == 1) for _ in 1:N) ≤ b

                @test a ≤ sum(Int(decide(cf, 1, -1e-4) == 2) for _ in 1:N) ≤ b
                @test a ≤ sum(Int(decide(cf, 2, -1e-4) == 1) for _ in 1:N) ≤ b

                a, b = cond(1.0)
                @test a ≤ sum(Int(decide(cf, 1, 1.0) == 2) for _ in 1:N) ≤ b
                @test a ≤ sum(Int(decide(cf, 2, 1.0) == 1) for _ in 1:N) ≤ b

                a, b = cond(0.0)
                @test a ≤ sum(Int(decide(cf, 1, -1.0) == 2) for _ in 1:N) ≤ b
                @test a ≤ sum(Int(decide(cf, 2, -1.0) == 1) for _ in 1:N) ≤ b
            end
        end
    end

    @testset "Multiple Agents" begin
        let cf = CounterFactual(Heaviside())
            @test decide(cf, [1, 2], [1., -1.]) == [2, 2]
            @test decide(cf, [1, 1], [1., -1.]) == [2, 1]
        end
    end

    @testset "From Payoffs" begin
        let cf = CounterFactual(Heaviside())
            @test_throws ArgumentError decide(cf, [0, 2], Float64[0 0; 0 0])
            @test_throws ArgumentError decide(cf, [3, 2], Float64[0 0; 0 0])
            @test_throws ArgumentError decide(cf, [1, 1], Float64[0 0 0; 0 0 0])
            @test_throws ArgumentError decide(cf, [1, 1, 1], Float64[0 0; 0 0])
            @test_throws ArgumentError decide(cf, [1, 1], Float64[0 0])
            @test_throws ArgumentError decide(cf, [1, 1], Float64[0 0; 0 0; 0 0])

            @test decide(cf, [1, 1], Float64[1 1; 0 0]) == [1, 1]
            @test decide(cf, [1, 1], Float64[1 0; 0 1]) == [1, 2]
            @test decide(cf, [1, 2], Float64[1 1; 0 0]) == [1, 1]
            @test decide(cf, [1, 2], Float64[1 0; 0 1]) == [1, 2]
            @test decide(cf, [2, 1], Float64[1 1; 0 0]) == [1, 1]
            @test decide(cf, [2, 1], Float64[1 0; 0 1]) == [1, 2]
            @test decide(cf, [2, 2], Float64[1 1; 0 0]) == [1, 1]
            @test decide(cf, [2, 2], Float64[1 0; 0 1]) == [1, 2]
        end
    end
end

@testset "Lattice Graph" begin
    @test_throws DomainError lattice_graph(0, 5)
    @test_throws DomainError lattice_graph(5, 0)
    @testset "Nodes and Edges" for nrows in 1:10, ncols in 1:10
        @test nv(lattice_graph(nrows, ncols)) == nrows * ncols
        @test ne(lattice_graph(nrows, ncols)) == 2 * nrows * ncols - nrows - ncols
    end
end

@testset "Arena" begin
    @testset "Construction" begin
        let game = Game(0, 1)
            scheme = CounterFactual()
            @test_throws DomainError Arena(game, SimpleGraph(), scheme)
            @test_throws DomainError Arena(game, SimpleGraph(1), scheme)
            let graph = SimpleGraph(2)
                add_edge!(graph, 1, 1)
                add_edge!(graph, 2, 2)
                @test_throws DomainError Arena(game, graph, scheme)
            end
            let graph = SimpleGraph(2)
                add_edge!(graph, 1, 1)
                add_edge!(graph, 1, 2)
                @test_throws DomainError Arena(game, graph, scheme)
            end
            let graph = SimpleGraph(3)
                add_edge!(graph, 1, 2)
                @test_throws DomainError Arena(game, graph, scheme)
            end

            @test length(Arena(game, cycle_graph(10), scheme)) == 10
        end
    end

    @testset "payoffs" begin
        let arena = Arena(Game(0, 1), path_graph(2), CounterFactual())
            @test_throws ArgumentError payoffs(arena, Int64[])
            @test_throws ArgumentError payoffs(arena, [1])
            @test_throws ArgumentError payoffs(arena, [1, 2, 1])
            @test_throws ArgumentError payoffs(arena, [0, 1])
            @test_throws ArgumentError payoffs(arena, [3, 1])

            @test payoffs(arena, [1, 1]) == [1 1; 1 1]
            @test payoffs(arena, [1, 2]) == [0 1; 0 1]
            @test payoffs(arena, [2, 1]) == [1 0; 1 0]
            @test payoffs(arena, [2, 2]) == [0 0; 0 0]
        end

        let arena = Arena(Game(0.25, 1.25), path_graph(2), CounterFactual())
            @test payoffs(arena, [1, 1]) == [1.00 1.00; 1.25 1.25]
            @test payoffs(arena, [1, 2]) == [0.25 1.00; 0.00 1.25]
        end

        let arena = Arena(Game(0.25, 1.25), lattice_graph(2, 2), CounterFactual())
            @test payoffs(arena, [1, 1, 1, 1]) == [2.00 2.00 2.00 2.00; 2.50 2.50 2.50 2.50]
            @test payoffs(arena, [1, 2, 1, 1]) == [1.25 2.00 2.00 1.25; 1.25 2.50 2.50 1.25]
            @test payoffs(arena, [2, 2, 2, 2]) == [0.50 0.50 0.50 0.50; 0.00 0.00 0.00 0.00]
        end
    end

    @testset "round" begin
        println("Random Seed: ", Random.GLOBAL_RNG.seed)
        Random.seed!(Random.GLOBAL_RNG.seed)

        let arena = Arena(Game(0.25, 1.25), lattice_graph(2, 2), CounterFactual(Heaviside()))
            N = 1000
            σ(p) = sqrt(N*p*(1-p))
            cond(p) = N*p - 3σ(p), N*p + 3σ(p)

            @test play(arena, [1, 1, 1, 1]) == [2, 2, 2, 2]
            @test play(arena, [2, 2, 2, 2]) == [1, 1, 1, 1]

            a, b = cond(0.5)
            dist = sum(Array{Int}(play(arena, [1, 2, 1, 1]) .== [1, 2, 2, 1]) for _ in 1:N)
            @test all([a, N, N, a] .<= dist .<= [b, N, N, b])
        end
    end
end
