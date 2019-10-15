using Gomen, Test, StaticArrays, Base.MathConstants, Random

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
