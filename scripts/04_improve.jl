# Step 6 of the loop: improvement. Better features, then a tree model via MLJ.
#   julia --project=. scripts/04_improve.jl
include(joinpath(@__DIR__, "..", "src", "Homestead.jl"))
using .Homestead, DataFrames, Printf, Statistics
import MLJ, MLJDecisionTreeInterface
using MLJ: machine, fit!

df  = add_features(clean(load_raw()))
s   = split_indices(nrow(df))
dev = vcat(s.train, s.val)
cvm(fit; kw...) = (c = cross_validate(fit, df, dev, 5; kw...); (mean(c), std(c)))

@printf "v1 numeric only, OLS        CV = %.4f ± %.4f\n" cvm(fit_ols)...
@printf "v2 + engineered + one-hot   OLS        CV = %.4f ± %.4f\n" cvm(fit_ols; featurizer = fit_features_v2)...
best_λ, best = 0.0, Inf
for λ in [1.0, 3.0, 10.0, 30.0, 100.0]
    m, sd = cvm((X, y) -> fit_ridge(X, y, λ); featurizer = fit_features_v2)
    @printf "v2                          Ridge λ=%-4.0f CV = %.4f ± %.4f\n" λ m sd
    global best_λ, best = m < best ? (λ, m) : (best_λ, best)
end

f = fit_features_v2(df, s.train)
X_train = design_matrix(f, df, s.train); y_train = df[s.train, "LogSalePrice"]
X_val   = design_matrix(f, df, s.val);   y_val   = df[s.val,   "LogSalePrice"]
println("v2 matrix: ", size(X_train, 2), " columns (", length(f.levels), " one-hot)")

m2 = fit_ridge(X_train, y_train, best_λ)
@printf "v2 Ridge λ=%.0f on validation  RMSE = %.4f  R² = %.3f\n" best_λ rmse(y_val, predict(m2, X_val)) r2(y_val, predict(m2, X_val))

names_ = feature_names(f)
nb = [(names_[i], m2.β[i]) for i in eachindex(names_) if startswith(names_[i], "Neighborhood=")]
sort!(nb, by = last)
println("Neighborhood offsets vs reference (NAmes), log price: lowest ", first(nb), "  highest ", last(nb))

# --- Random forest via MLJ ---------------------------------------------------
RF = MLJ.@load RandomForestRegressor pkg = DecisionTree verbosity = 0
Xt = MLJ.table(X_train[:, 2:end]); Xv = MLJ.table(X_val[:, 2:end])
rf = machine(RF(n_trees = 300, max_depth = -1, min_samples_leaf = 2, n_subfeatures = -1, rng = 42), Xt, y_train)
fit!(rf, verbosity = 0)
ŷ_rf = MLJ.predict(rf, Xv)
@printf "Random forest (300 trees)   validation RMSE = %.4f  R² = %.3f\n" rmse(y_val, ŷ_rf) r2(y_val, ŷ_rf)

# Blend: average the two in log space
ŷ_blend = (predict(m2, X_val) .+ ŷ_rf) ./ 2
@printf "Blend ridge+forest          validation RMSE = %.4f  R² = %.3f\n" rmse(y_val, ŷ_blend) r2(y_val, ŷ_blend)

println("\nWorst validation misses, v2 ridge:")
show(worst_residuals(df, s.val, predict(m2, X_val); top = 5), allrows = true, allcols = true); println()
