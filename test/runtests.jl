using Gomen, Test, StaticArrays

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
