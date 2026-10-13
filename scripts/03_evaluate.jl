# Step 5 of the loop: evaluation. 5-fold CV, a ridge λ sweep, and error analysis.
#   julia --project=. scripts/03_evaluate.jl
include(joinpath(@__DIR__, "..", "src", "Homestead.jl"))
using .Homestead, DataFrames, Printf, Statistics

df = clean(load_raw())
s  = split_indices(nrow(df))
dev = vcat(s.train, s.val)          # CV uses train+val; test stays untouched

println("5-fold CV on ", length(dev), " rows (test set still held out)")
cv = cross_validate(fit_ols, df, dev, 5)
@printf "OLS      folds = %s  mean = %.4f  std = %.4f\n" join([@sprintf("%.4f", x) for x in cv], " ") mean(cv) std(cv)

println("\nRidge λ sweep (5-fold CV mean RMSE):")
best_λ, best = 0.0, Inf
for λ in [0.0, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0]
    m = mean(cross_validate((X, y) -> fit_ridge(X, y, λ), df, dev, 5))
    @printf "  λ = %6.1f   %.4f%s\n" λ m (m < best ? "  ←" : "")
    global best_λ, best = m < best ? (λ, m) : (best_λ, best)
end
@printf "Best λ = %.1f\n" best_λ

println("\nWorst predictions on the validation split (OLS):")
cols = numeric_features(df); st = fit_standardizer(df, s.train, cols)
m = fit_ols(design_matrix(st, df, s.train), df[s.train, "LogSalePrice"])
ŷ = predict(m, design_matrix(st, df, s.val))
show(worst_residuals(df, s.val, ŷ; top = 8), allrows = true, allcols = true); println()

r = df[s.val, "LogSalePrice"] .- ŷ
@printf "\nResiduals: mean = %.4f  (should be ≈ 0)   share within ±10%%: %.0f%%   within ±20%%: %.0f%%\n" mean(r) 100mean(abs.(r) .< log(1.1)) 100mean(abs.(r) .< log(1.2))
