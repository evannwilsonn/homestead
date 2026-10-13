# Self-improving replay: 2006–2010 sales arrive one month at a time.
#   julia --project=. scripts/06_replay.jl
include(joinpath(@__DIR__, "..", "src", "Homestead.jl"))
using .Homestead, DataFrames, Printf, Statistics, CSV, JSON3

df = clean(load_raw())
res, promos, bank, final = replay(df)

out = DataFrame(month = [r.month for r in res], n = [r.n for r in res],
                rmse_frozen = [r.rmse_frozen for r in res], rmse_online = [r.rmse_online for r in res],
                rmse_system = [r.rmse_system for r in res], features = [r.recipe_features for r in res],
                lambda = [r.λ for r in res], promoted = [r.promoted for r in res], drift = [r.drift for r in res])
CSV.write(joinpath(@__DIR__, "..", "results", "replay.csv"), out)
open(joinpath(@__DIR__, "..", "results", "promotions.json"), "w") do io; JSON3.pretty(io, promos); end

# Pooled prequential RMSE (every post-warm-up sale scored once, before it was learned)
pooled(col) = sqrt(sum(out.n .* out[!, col] .^ 2) / sum(out.n))
println()
for (name, col) in [("frozen", :rmse_frozen), ("online", :rmse_online), ("system", :rmse_system)]
    @printf "%-7s pooled RMSE %.4f  ≈ ±%.1f%%   first year %.4f   last year %.4f\n" name pooled(col) pct_error(pooled(col)) mean(out[1:12, col]) mean(out[end-11:end, col])
end
println("promotions: ", length(promos), "   drift months: ", join(out.month[out.drift], ", "))
println("final recipe: ", describe_recipe(bank, final.recipe))
