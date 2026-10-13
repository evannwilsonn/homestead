"""
Save a fitted v2 ridge model to JSON and predict from raw house fields.

The JSON holds everything the endpoint needs with no DataFrame of training
data around: the column list, standardization means/stds, one-hot levels,
the coefficient vector, and a default value for every raw input column (the
training median for numbers, the most common level for strings) so a request
may omit fields it does not know.
"""

using JSON3, DataFrames, Statistics

struct SavedModel
    cols::Vector{String}          # numeric columns in X order (after add_features)
    μ::Vector{Float64}
    σ::Vector{Float64}
    levels::Vector{Tuple{String,String}}
    β::Vector{Float64}
    λ::Float64
    defaults::Dict{String,Any}    # raw column → default value
    metrics::Dict{String,Float64}
end

"Most common value in a vector."
mode_of(v) = (c = Dict{Any,Int}(); for x in v; c[x] = get(c, x, 0) + 1; end; argmax(c))

function SavedModel(f::FeaturesV2, m::LinearModel, λ, raw::DataFrame, rows, metrics)
    defaults = Dict{String,Any}()
    for c in names(raw)
        c in TARGETS && continue
        v = raw[rows, c]
        defaults[c] = eltype(v) <: Number ? median(v) : mode_of(v)
    end
    SavedModel(f.numeric.cols, f.numeric.μ, f.numeric.σ, f.levels, m.β, Float64(λ), defaults, metrics)
end

function save_model(path::AbstractString, sm::SavedModel)
    open(path, "w") do io
        JSON3.pretty(io, Dict(
            "cols" => sm.cols, "mu" => sm.μ, "sigma" => sm.σ,
            "levels" => [[c, l] for (c, l) in sm.levels],
            "beta" => sm.β, "lambda" => sm.λ, "defaults" => sm.defaults, "metrics" => sm.metrics))
    end
    path
end

function load_model(path::AbstractString)
    j = JSON3.read(read(path, String))
    SavedModel(String.(j.cols), Float64.(j.mu), Float64.(j.sigma),
               [(String(p[1]), String(p[2])) for p in j.levels],
               Float64.(j.beta), Float64(j.lambda),
               Dict{String,Any}(String(k) => v for (k, v) in pairs(j.defaults)),
               Dict{String,Float64}(String(k) => Float64(v) for (k, v) in pairs(j.metrics)))
end

"""
predict_price(sm, house::Dict) → (price, log_price)

`house` maps raw Ames column names (e.g. "Gr Liv Area", "Neighborhood") to
values. Missing columns take the training default. Engineered columns are
rebuilt with `add_features`, so the request never has to supply them.
"""
function predict_price(sm::SavedModel, house::AbstractDict)
    row = Dict{String,Any}(k => get(house, k, v) for (k, v) in sm.defaults)
    df = DataFrame([k => [v] for (k, v) in row]...)
    for c in keys(sm.defaults)                      # keep raw column types stable
        if sm.defaults[c] isa Number
            df[!, c] = Float64.(df[!, c])
        else
            df[!, c] = String.(df[!, c])
        end
    end
    df = add_features(df)
    x = [1.0; (Float64.([df[1, c] for c in sm.cols]) .- sm.μ) ./ sm.σ;
         [Float64(df[1, c] == l) for (c, l) in sm.levels]]
    lp = x' * sm.β
    (price = exp(lp), log_price = lp)
end
