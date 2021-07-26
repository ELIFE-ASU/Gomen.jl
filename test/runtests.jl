using Gomen
using Base.MathConstants, Random, Statistics, Test
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

@testset "Games" begin
    @testset "length" begin
        @test length(Games(0.5, 0.5)) == 9
        @test length(Games(0.1, 0.1)) == 121
        @test length(Games(0.01, 0.01)) == 10201
        @test length(Games(0.3, 0.3)) == 16
    end

    @testset "collect" begin
        @test map(g -> g.payoffs, Games(1, 1)) == [
                    SMatrix{2,2}([1.0 -0.5; 0.5 0.0]),
                    SMatrix{2,2}([1.0  0.5; 0.5 0.0]),
                    SMatrix{2,2}([1.0 -0.5; 1.5 0.0]),
                    SMatrix{2,2}([1.0  0.5; 1.5 0.0])
                ]

        @test map(g -> g.payoffs, Games(0.5, 1)) == [
                    SMatrix{2,2}([1.0 -0.5; 0.5 0.0]),
                    SMatrix{2,2}([1.0  0.0; 0.5 0.0]),
                    SMatrix{2,2}([1.0  0.5; 0.5 0.0]),
                    SMatrix{2,2}([1.0 -0.5; 1.5 0.0]),
                    SMatrix{2,2}([1.0  0.0; 1.5 0.0]),
                    SMatrix{2,2}([1.0  0.5; 1.5 0.0])
                ]

        @test map(g -> g.payoffs, Games(1, 0.5)) == [
                    SMatrix{2,2}([1.0 -0.5; 0.5 0.0]),
                    SMatrix{2,2}([1.0  0.5; 0.5 0.0]),
                    SMatrix{2,2}([1.0 -0.5; 1.0 0.0]),
                    SMatrix{2,2}([1.0  0.5; 1.0 0.0]),
                    SMatrix{2,2}([1.0 -0.5; 1.5 0.0]),
                    SMatrix{2,2}([1.0  0.5; 1.5 0.0])
                ]

        @test map(g -> g.payoffs, Games(0.5, 0.5)) == [
                    SMatrix{2,2}([1.0 -0.5; 0.5 0.0]),
                    SMatrix{2,2}([1.0  0.0; 0.5 0.0]),
                    SMatrix{2,2}([1.0  0.5; 0.5 0.0]),
                    SMatrix{2,2}([1.0 -0.5; 1.0 0.0]),
                    SMatrix{2,2}([1.0  0.0; 1.0 0.0]),
                    SMatrix{2,2}([1.0  0.5; 1.0 0.0]),
                    SMatrix{2,2}([1.0 -0.5; 1.5 0.0]),
                    SMatrix{2,2}([1.0  0.0; 1.5 0.0]),
                    SMatrix{2,2}([1.0  0.5; 1.5 0.0])
                ]

        @test map(g -> g.payoffs, Games(0.3, 0.3)) == [
                    SMatrix{2,2}([1.0 -0.5; 0.5 0.0]),
                    SMatrix{2,2}([1.0 -0.2; 0.5 0.0]),
                    SMatrix{2,2}([1.0  0.1; 0.5 0.0]),
                    SMatrix{2,2}([1.0  0.4; 0.5 0.0]),
                    SMatrix{2,2}([1.0 -0.5; 0.8 0.0]),
                    SMatrix{2,2}([1.0 -0.2; 0.8 0.0]),
                    SMatrix{2,2}([1.0  0.1; 0.8 0.0]),
                    SMatrix{2,2}([1.0  0.4; 0.8 0.0]),
                    SMatrix{2,2}([1.0 -0.5; 1.1 0.0]),
                    SMatrix{2,2}([1.0 -0.2; 1.1 0.0]),
                    SMatrix{2,2}([1.0  0.1; 1.1 0.0]),
                    SMatrix{2,2}([1.0  0.4; 1.1 0.0]),
                    SMatrix{2,2}([1.0 -0.5; 1.4 0.0]),
                    SMatrix{2,2}([1.0 -0.2; 1.4 0.0]),
                    SMatrix{2,2}([1.0  0.1; 1.4 0.0]),
                    SMatrix{2,2}([1.0  0.4; 1.4 0.0])
                ]
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
    game = Game(0.3, 1.3)
    graph = first(CycleGraphGenerator(1,3))
    @testset "Single Agent" begin
        let N = 1000
            σ(p) = sqrt(N*p*(1-p))
            cond(p) = N*p - 3σ(p), N*p + 3σ(p)

            seed = Random.rand(UInt32)
            println("Random Seed: ", seed)
            Random.seed!(seed)

            let cf = CounterFactual()
                arena = Arena(game, graph, cf)

                a, b = cond(0.5)
                @test a ≤ sum(Int(decide(cf, arena, 1, 0.0) == 2) for _ in 1:N) ≤ b
                @test a ≤ sum(Int(decide(cf, arena, 2, 0.0) == 1) for _ in 1:N) ≤ b

                a, b = cond(e / (1 + e))
                @test a ≤ sum(Int(decide(cf, arena, 1, 1.0) == 2) for _ in 1:N) ≤ b
                @test a ≤ sum(Int(decide(cf, arena, 2, 1.0) == 1) for _ in 1:N) ≤ b

                a, b = cond(1 / (1 + e))
                @test a ≤ sum(Int(decide(cf, arena, 1, -1.0) == 2) for _ in 1:N) ≤ b
                @test a ≤ sum(Int(decide(cf, arena, 2, -1.0) == 1) for _ in 1:N) ≤ b
            end

            let cf = CounterFactual(Sigmoid(0.5))
                arena = Arena(game, graph, cf)

                a, b = cond(0.5)
                @test a ≤ sum(Int(decide(cf, arena, 1, 0.0) == 2) for _ in 1:N) ≤ b
                @test a ≤ sum(Int(decide(cf, arena, 2, 0.0) == 1) for _ in 1:N) ≤ b

                a, b = cond(sqrt(e) / (1 + sqrt(e)))
                @test a ≤ sum(Int(decide(cf, arena, 1, 1.0) == 2) for _ in 1:N) ≤ b
                @test a ≤ sum(Int(decide(cf, arena, 2, 1.0) == 1) for _ in 1:N) ≤ b

                a, b = cond(1 / (1 + sqrt(e)))
                @test a ≤ sum(Int(decide(cf, arena, 1, -1.0) == 2) for _ in 1:N) ≤ b
                @test a ≤ sum(Int(decide(cf, arena, 2, -1.0) == 1) for _ in 1:N) ≤ b
            end

            let cf = CounterFactual(Heaviside())
                arena = Arena(game, graph, cf)

                a, b = cond(0.5)
                @test a ≤ sum(Int(decide(cf, arena, 1, 0.0) == 2) for _ in 1:N) ≤ b
                @test a ≤ sum(Int(decide(cf, arena, 2, 0.0) == 1) for _ in 1:N) ≤ b

                @test a ≤ sum(Int(decide(cf, arena, 1, 1e-4) == 2) for _ in 1:N) ≤ b
                @test a ≤ sum(Int(decide(cf, arena, 2, 1e-4) == 1) for _ in 1:N) ≤ b

                @test a ≤ sum(Int(decide(cf, arena, 1, -1e-4) == 2) for _ in 1:N) ≤ b
                @test a ≤ sum(Int(decide(cf, arena, 2, -1e-4) == 1) for _ in 1:N) ≤ b

                a, b = cond(1.0)
                @test a ≤ sum(Int(decide(cf, arena, 1, 1.0) == 2) for _ in 1:N) ≤ b
                @test a ≤ sum(Int(decide(cf, arena, 2, 1.0) == 1) for _ in 1:N) ≤ b

                a, b = cond(0.0)
                @test a ≤ sum(Int(decide(cf, arena, 1, -1.0) == 2) for _ in 1:N) ≤ b
                @test a ≤ sum(Int(decide(cf, arena, 2, -1.0) == 1) for _ in 1:N) ≤ b
            end
        end
    end

    @testset "Multiple Agents" begin
        let cf = CounterFactual(Heaviside())
            arena = Arena(game, graph, cf)

            @test decide(cf, arena, [1, 2], [1., -1.]) == [2, 2]
            @test decide(cf, arena, [1, 1], [1., -1.]) == [2, 1]
        end
    end

    @testset "From Payoffs" begin
        let cf = CounterFactual(Heaviside())
            arena = Arena(game, graph, cf)

            @test_throws ArgumentError decide(cf, arena, [0, 2], Float64[0 0; 0 0])
            @test_throws ArgumentError decide(cf, arena, [3, 2], Float64[0 0; 0 0])
            @test_throws ArgumentError decide(cf, arena, [1, 1], Float64[0 0 0; 0 0 0])
            @test_throws ArgumentError decide(cf, arena, [1, 1, 1], Float64[0 0; 0 0])
            @test_throws ArgumentError decide(cf, arena, [1, 1], Float64[0 0])
            @test_throws ArgumentError decide(cf, arena, [1, 1], Float64[0 0; 0 0; 0 0])

            @test decide(cf, arena, [1, 1], Float64[1 1; 0 0]) == [1, 1]
            @test decide(cf, arena, [1, 1], Float64[1 0; 0 1]) == [1, 2]
            @test decide(cf, arena, [1, 2], Float64[1 1; 0 0]) == [1, 1]
            @test decide(cf, arena, [1, 2], Float64[1 0; 0 1]) == [1, 2]
            @test decide(cf, arena, [2, 1], Float64[1 1; 0 0]) == [1, 1]
            @test decide(cf, arena, [2, 1], Float64[1 0; 0 1]) == [1, 2]
            @test decide(cf, arena, [2, 2], Float64[1 1; 0 0]) == [1, 1]
            @test decide(cf, arena, [2, 2], Float64[1 0; 0 1]) == [1, 2]
        end
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
            @test_throws ArgumentError payoffs(arena, Int[])
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

        let game = Game(0.25, 1.25)
            graph = first(GridGraphGenerator(; count=1, dims=[2,2]))
            scheme = CounterFactual(Heaviside())
            arena = Arena(game, graph, scheme)

            @test payoffs(arena, [1, 1, 1, 1]) == [2.00 2.00 2.00 2.00; 2.50 2.50 2.50 2.50]
            @test payoffs(arena, [1, 2, 1, 1]) == [1.25 2.00 2.00 1.25; 1.25 2.50 2.50 1.25]
            @test payoffs(arena, [2, 2, 2, 2]) == [0.50 0.50 0.50 0.50; 0.00 0.00 0.00 0.00]
        end
    end

    @testset "round" begin
        seed = Random.rand(UInt32)
        println("Random Seed: ", seed)
        Random.seed!(seed)

        let game = Game(0.25, 1.25)
            graph = first(GridGraphGenerator(; count=1, dims=[2,2]))
            scheme = CounterFactual(Heaviside())
            arena = Arena(game, graph, scheme)

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

@testset "ROC" begin
    @testset "Construction" begin
        @test fpr(ROC()) == [0.0, 1.0]
        @test tpr(ROC()) == [0.0, 1.0]

        @test_throws DimensionMismatch ROC([0.5], Float64[])
        @test_throws DimensionMismatch ROC(Float64[], [0.5])

        @test_throws ArgumentError ROC([0.1, 0.2], [0.5, 0.4])
        @test_throws ArgumentError ROC([0.2, 0.1], [0.4, 0.5])

        @test_throws ArgumentError ROC([-0.5, 0.5], [0.5, 0.8])
        @test_throws ArgumentError ROC([0.5, -0.5], [0.8, 0.5])
        @test_throws ArgumentError ROC([0.5, 0.8], [-0.5, 0.5])
        @test_throws ArgumentError ROC([0.8, 0.5], [0.5, -0.5])

        @test_throws ArgumentError ROC([0.5, 2.0], [0.5, 0.8])
        @test_throws ArgumentError ROC([2.0, 0.5], [0.8, 0.5])
        @test_throws ArgumentError ROC([0.5, 0.8], [0.5, 2.0])
        @test_throws ArgumentError ROC([0.8, 0.5], [2.0, 0.5])

        let r = ROC([0.3, 0.8], [0.5, 0.9])
            @test length(fpr(r)) == length(tpr(r)) == 4
            @test fpr(r)[1] ≈ tpr(r)[1] ≈ 0.0
            @test fpr(r)[end] ≈ tpr(r)[end] ≈ 1.0
            @test issorted(fpr(r))
            @test issorted(tpr(r))
        end

        let r = ROC([0.8, 0.3], [0.9, 0.5])
            @test fpr(r) ≈ [0.0, 0.3, 0.8, 1.0]
            @test tpr(r) ≈ [0.0, 0.5, 0.9, 1.0]
        end

        let r = ROC([0.3, 0.8, 0.3], [0.5, 0.9, 0.5])
            @test fpr(r) ≈ [0.0, 0.3, 0.8, 1.0]
            @test tpr(r) ≈ [0.0, 0.5, 0.9, 1.0]
        end

        let r = ROC([0.3, 0.8, 0.3, 0.0], [0.5, 0.9, 0.5, 0.2])
            @test fpr(r) ≈ [0.0, 0.0, 0.3, 0.8, 1.0]
            @test tpr(r) ≈ [0.0, 0.2, 0.5, 0.9, 1.0]
        end

        let r = ROC([0.3, 0.8, 0.3, 0.2], [0.5, 0.9, 0.5, 0.0])
            @test fpr(r) ≈ [0.0, 0.2, 0.3, 0.8, 1.0]
            @test tpr(r) ≈ [0.0, 0.0, 0.5, 0.9, 1.0]
        end

        let r = ROC([0.3, 0.8, 0.3, 0.0], [0.5, 0.9, 0.5, 0.0])
            @test fpr(r) ≈ [0.0, 0.3, 0.8, 1.0]
            @test tpr(r) ≈ [0.0, 0.5, 0.9, 1.0]
        end

        let r = ROC([0.3, 1.0, 0.8, 0.3], [0.5, 0.95, 0.9, 0.5])
            @test fpr(r) ≈ [0.0, 0.3, 0.8, 1.0, 1.0]
            @test tpr(r) ≈ [0.0, 0.5, 0.9, 0.95, 1.0]
        end

        let r = ROC([0.3, 0.95, 0.8, 0.3], [0.5, 1.0, 0.9, 0.5])
            @test fpr(r) ≈ [0.0, 0.3, 0.8, 0.95, 1.0]
            @test tpr(r) ≈ [0.0, 0.5, 0.9, 1.0, 1.0]
        end

        let r = ROC([0.3, 1.0, 0.8, 0.3], [0.5, 1.0, 0.9, 0.5])
            @test fpr(r) ≈ [0.0, 0.3, 0.8, 1.0]
            @test tpr(r) ≈ [0.0, 0.5, 0.9, 1.0]
        end

        let r = ROC([0.3, 0.3, 0.4, 0.4], [0.4, 0.2, 0.8, 0.6])
            @test fpr(r) ≈ [0.0, 0.3, 0.3, 0.4, 0.4, 1.0]
            @test tpr(r) ≈ [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        end
    end

    @testset "mean" begin
        let r = ROC(), s = ROC()
            μ = mean([r, s])
            @test fpr(μ) ≈ [0.0, 1.0]
            @test tpr(μ) ≈ [0.0, 1.0]
        end

        let r = ROC(), s = ROC([0.5], [0.5])
            μ1, μ2 = mean([r, s]), mean([s, r])
            @test fpr(μ1) ≈ fpr(μ2) ≈ [0.0, 0.5, 1.0]
            @test tpr(μ1) ≈ tpr(μ2) ≈ [0.0, 0.5, 1.0]
        end

        let r = ROC(), s = ROC([0.5], [0.0])
            μ1, μ2 = mean([r, s]), mean([s, r])
            @test fpr(μ1) ≈ fpr(μ2) ≈ [0.0, 0.5, 1.0]
            @test tpr(μ1) ≈ tpr(μ2) ≈ [0.0, 0.25, 1.0]
        end

        let r = ROC([0.5], [0.0]), s = ROC([0.5], [0.5])
            μ1, μ2 = mean([r, s]), mean([s, r])
            @test fpr(μ1) ≈ fpr(μ2) ≈ [0.0, 0.5, 1.0]
            @test tpr(μ1) ≈ tpr(μ2) ≈ [0.0, 0.25, 1.0]
        end

        let r = ROC([0.25, 0.75], [0.3, 0.8]), s = ROC([0.5], [0.25])
            μ1, μ2 = mean([r, s]), mean([s, r])
            @test fpr(μ1) ≈ fpr(μ2) ≈ [0.0, 0.25, 0.5, 0.75, 1.0]
            @test tpr(μ1) ≈ tpr(μ2) ≈ [0.0, 0.2125, 0.4, 0.7125, 1.0]
        end
    end

    @testset "AUC" begin
        @test auc(ROC()) ≈ 0.5
        @test auc(ROC([0.5], [0.5])) ≈ 0.5
        @test auc(ROC([0.25, 0.6], [0.25, 0.6])) ≈ 0.5

        @test auc(ROC([0.5], [0.0])) ≈ 0.25
        @test auc(ROC([0.5], [1.0])) ≈ 0.75

        @test auc(ROC([0.2, 0.3, 0.8, 0.9], [0.1, 0.4, 0.85, 0.95])) ≈ 0.5350
    end
end
