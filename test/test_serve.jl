# Integration test: start the server, hit it, stop it.
#   julia --project=. test/test_serve.jl
using Test, HTTP, JSON3
include(joinpath(@__DIR__, "..", "src", "serve.jl"))

server = start(port = 8099)
try
    @testset "HTTP endpoint" begin
        r = HTTP.get("http://127.0.0.1:8099/")
        @test r.status == 200
        meta = JSON3.read(String(r.body))
        @test meta.coefficients == 116
        @test 0.05 < meta.metrics.test_rmse_log < 0.2

        body = JSON3.write(Dict("Gr Liv Area" => 1500, "Overall Qual" => 7, "Year Built" => 2005, "Neighborhood" => "CollgCr"))
        r = HTTP.post("http://127.0.0.1:8099/predict", ["Content-Type" => "application/json"], body)
        @test r.status == 200
        p = JSON3.read(String(r.body))
        @test 100_000 < p.price < 300_000

        better = JSON3.write(Dict("Gr Liv Area" => 2500, "Overall Qual" => 9, "Year Built" => 2008, "Neighborhood" => "NridgHt"))
        p2 = JSON3.read(String(HTTP.post("http://127.0.0.1:8099/predict", ["Content-Type" => "application/json"], better).body))
        @test p2.price > p.price                    # bigger, newer, nicer costs more

        r = HTTP.post("http://127.0.0.1:8099/predict", ["Content-Type" => "application/json"], "not json"; status_exception = false)
        @test r.status == 400
        @test HTTP.get("http://127.0.0.1:8099/nope"; status_exception = false).status == 404
    end
finally
    close(server)
end
