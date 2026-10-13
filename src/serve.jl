"""
Tiny HTTP API around the saved model.

    julia --project=. src/serve.jl            # listens on 127.0.0.1:8080

    GET  /            → model metadata (test metrics, coefficient count)
    POST /predict     → {"price": 177555.0, "log_price": 12.087}
                        body: JSON object of Ames column names → values,
                        e.g. {"Gr Liv Area": 1500, "Overall Qual": 7, "Neighborhood": "CollgCr"}
                        Unknown fields are ignored; missing ones use training defaults.
"""

include(joinpath(@__DIR__, "Homestead.jl"))
using .Homestead, HTTP, JSON3

const MODEL_PATH = joinpath(@__DIR__, "..", "model", "homestead_v2.json")
const MODEL = Ref{SavedModel}()

json(status, obj) = HTTP.Response(status, ["Content-Type" => "application/json"], JSON3.write(obj))

function handle(req::HTTP.Request)
    try
        if req.method == "GET" && req.target == "/"
            return json(200, Dict("model" => "homestead_v2 ridge", "coefficients" => length(MODEL[].β),
                                  "lambda" => MODEL[].λ, "metrics" => MODEL[].metrics))
        elseif req.method == "POST" && req.target == "/predict"
            body = JSON3.read(String(req.body))
            house = Dict{String,Any}(String(k) => v for (k, v) in pairs(body))
            p = predict_price(MODEL[], house)
            return json(200, Dict("price" => round(p.price), "log_price" => p.log_price))
        else
            return json(404, Dict("error" => "use GET / or POST /predict"))
        end
    catch e
        return json(400, Dict("error" => sprint(showerror, e)))
    end
end

function start(; host = "127.0.0.1", port = 8080, async = false)
    MODEL[] = load_model(MODEL_PATH)
    predict_price(MODEL[], Dict{String,Any}())   # warm-up: compile the predict path before the first request
    @info "homestead serving on http://$host:$port  ($(length(MODEL[].β)) coefficients)"
    HTTP.serve!(handle, host, port)
end

if abspath(PROGRAM_FILE) == @__FILE__
    server = start()
    wait(server)
end
