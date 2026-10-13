# Step 4 of the loop: linear models from scratch, checked against MLJ.
#   julia --project=. scripts/02_linear.jl
include(joinpath(@__DIR__, "..", "src", "Homestead.jl"))
using .Homestead, DataFrames, Printf, LinearAlgebra
import MLJ, MLJLinearModels
using MLJ: machine, fit!, fitted_params

df = clean(load_raw())
s  = split_indices(nrow(df))
cols = numeric_features(df)
st = fit_standardizer(df, s.train, cols)
X_train = design_matrix(st, df, s.train);  y_train = df[s.train, "LogSalePrice"]
X_val   = design_matrix(st, df, s.val);    y_val   = df[s.val,   "LogSalePrice"]
println("Features: ", length(cols), " numeric columns + intercept, ", size(X_train, 1), " training rows")

show_score(name, m) = @printf "%-28s RMSE(log) = %.4f  ≈ ±%.1f%%  R² = %.3f\n" name rmse(y_val, predict(m, X_val)) pct_error(rmse(y_val, predict(m, X_val))) r2(y_val, predict(m, X_val))

ols_n = fit_ols_normal(X_train, y_train);  show_score("OLS (normal equation)", ols_n)
ols   = fit_ols(X_train, y_train);         show_score("OLS (QR solve)", ols)
@printf "  max |β_normal − β_qr| = %.2e\n" maximum(abs.(ols_n.β .- ols.β))

ridge = fit_ridge(X_train, y_train, 10.0); show_score("Ridge λ=10", ridge)
gd, hist = fit_gd(X_train, y_train; lr = 0.1, epochs = 2000); show_score("Gradient descent 2000 ep", gd)
@printf "  GD loss: epoch 1 = %.4f, epoch 100 = %.4f, epoch 2000 = %.4f;  max |β_gd − β_ols| = %.2e\n" hist[1] hist[100] hist[end] maximum(abs.(gd.β .- ols.β))

# --- MLJ check: same features, library implementation -----------------------
Xt = MLJ.table(X_train[:, 2:end]);  Xv = MLJ.table(X_val[:, 2:end])
lr_mach = machine(MLJLinearModels.LinearRegressor(fit_intercept = true), Xt, y_train) |> fit!
ŷ_mlj = MLJ.predict(lr_mach, Xv)
@printf "%-28s RMSE(log) = %.4f\n" "MLJ LinearRegressor" rmse(y_val, ŷ_mlj)
fp = fitted_params(lr_mach)
β_mlj = vcat(fp.intercept, [v for (_, v) in fp.coefs])
@printf "  max |β_mlj − β_ols| = %.2e\n" maximum(abs.(β_mlj .- ols.β))

rg_mach = machine(MLJLinearModels.RidgeRegressor(lambda = 10.0, fit_intercept = true, scale_penalty_with_samples = false), Xt, y_train) |> fit!
@printf "%-28s RMSE(log) = %.4f\n" "MLJ RidgeRegressor λ=10" rmse(y_val, MLJ.predict(rg_mach, Xv))
