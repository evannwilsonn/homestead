# Step 7a: fit the final model on train+val, save it, score the TEST set once.
#   julia --project=. scripts/05_train_final.jl
include(joinpath(@__DIR__, "..", "src", "Homestead.jl"))
using .Homestead, DataFrames, Printf, Statistics

const λ = 30.0
raw = clean(load_raw())
df  = add_features(raw)
s   = split_indices(nrow(df))
dev = vcat(s.train, s.val)

f = fit_features_v2(df, dev)
m = fit_ridge(design_matrix(f, df, dev), df[dev, "LogSalePrice"], λ)

# The one and only look at the test set.
ŷ = predict(m, design_matrix(f, df, s.test)); y = df[s.test, "LogSalePrice"]
metrics = Dict("test_rmse_log" => rmse(y, ŷ), "test_r2" => r2(y, ŷ), "test_mae_usd" => mean(abs.(exp.(y) .- exp.(ŷ))), "n_test" => Float64(length(y)))
@printf "TEST (%d houses, touched once): RMSE(log) = %.4f  ≈ ±%.1f%%   R² = %.3f   MAE = \$%.0f\n" length(y) metrics["test_rmse_log"] pct_error(metrics["test_rmse_log"]) metrics["test_r2"] metrics["test_mae_usd"]

path = save_model(joinpath(@__DIR__, "..", "model", "homestead_v2.json"), SavedModel(f, m, λ, raw, dev, metrics))
println("saved ", path, "  (", length(m.β), " coefficients)")

sm = load_model(path)
ex = Dict("Gr Liv Area" => 1500, "Overall Qual" => 7, "Year Built" => 2005, "Yr Sold" => 2010,
          "Neighborhood" => "CollgCr", "Total Bsmt SF" => 1000, "Garage Cars" => 2, "Full Bath" => 2, "Lot Area" => 9000)
@printf "example: 1500 sqft, qual 7, built 2005, CollgCr → \$%.0f\n" predict_price(sm, ex).price
