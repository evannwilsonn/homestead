### A Pluto.jl notebook ###
# v0.20.0

using Markdown
using InteractiveUtils

# ╔═╡ 1a000000-0000-4000-8000-000000000001
begin
    import Pkg
    Pkg.activate(joinpath(@__DIR__, ".."))
    include(joinpath(@__DIR__, "..", "src", "Homestead.jl"))
    using .Homestead, DataFrames, Statistics, Printf
end

# ╔═╡ 1a000000-0000-4000-8000-000000000002
md"""
# Homestead: predicting Ames house prices

This notebook walks the loop from the project README:
**Problem → Data → Baseline → Model → Evaluation → Improvement → Deployment.**

Every number here is recomputed when the notebook runs. Change the λ slider or
the house below and the rest updates.
"""

# ╔═╡ 1a000000-0000-4000-8000-000000000003
md"""
## 1. Problem
Predict a house's sale price from its description. We model **log(price)**,
so errors are relative: a 10% miss counts the same on a \$100k house and a \$500k house.
Metric: RMSE on log price (0.10 ≈ ±10%).
"""

# ╔═╡ 1a000000-0000-4000-8000-000000000004
begin
    raw = clean(load_raw())
    df  = add_features(raw)
    s   = split_indices(nrow(df))
    (rows = nrow(df), columns = ncol(df), train = length(s.train), val = length(s.val), test = length(s.test))
end

# ╔═╡ 1a000000-0000-4000-8000-000000000005
md"""
## 2. Data
2,930 houses sold in Ames, Iowa, 2006–2010, with 80 features (De Cock 2011).
Five >4,000 sq ft partial sales are removed as the paper recommends.
Split 70 / 15 / 15. The test rows are not used anywhere below until the last cell.
"""

# ╔═╡ 1a000000-0000-4000-8000-000000000006
describe(df[:, ["SalePrice", "Gr Liv Area", "Overall Qual", "Year Built", "Lot Area"]], :min, :median, :max)

# ╔═╡ 1a000000-0000-4000-8000-000000000007
md"## 3. Baseline"

# ╔═╡ 1a000000-0000-4000-8000-000000000008
begin
    y_tr = df[s.train, "LogSalePrice"]; y_va = df[s.val, "LogSalePrice"]
    med  = fit_median(y_tr)
    line = fit_one_feature(Float64.(df[s.train, "Gr Liv Area"]), y_tr)
    (median_rmse = rmse(y_va, predict(med, length(y_va))),
     living_area_line_rmse = rmse(y_va, predict(line, Float64.(df[s.val, "Gr Liv Area"]))))
end

# ╔═╡ 1a000000-0000-4000-8000-000000000009
md"""
## 4. Model: linear regression three ways, from scratch

The normal equation `β = (XᵀX)⁻¹Xᵀy`, a QR solve of the same system, and
gradient descent all fit the same model. They should agree.
"""

# ╔═╡ 1a000000-0000-4000-8000-00000000000a
begin
    st1 = fit_standardizer(df, s.train, numeric_features(df))
    X1_tr = design_matrix(st1, df, s.train); X1_va = design_matrix(st1, df, s.val)
    ols_n = fit_ols_normal(X1_tr, y_tr); ols_q = fit_ols(X1_tr, y_tr)
    gd, loss = fit_gd(X1_tr, y_tr; lr = 0.1, epochs = 1500)
    (rmse_normal_eq = rmse(y_va, predict(ols_n, X1_va)),
     rmse_qr = rmse(y_va, predict(ols_q, X1_va)),
     rmse_gd = rmse(y_va, predict(gd, X1_va)),
     max_coef_gap_normal_vs_qr = maximum(abs.(ols_n.β .- ols_q.β)),
     max_coef_gap_gd_vs_qr = maximum(abs.(gd.β .- ols_q.β)))
end

# ╔═╡ 1a000000-0000-4000-8000-00000000000b
md"Gradient-descent loss by epoch (first 60 shown; it keeps falling, slowly):"

# ╔═╡ 1a000000-0000-4000-8000-00000000000c
round.(loss[1:60]; digits = 4)

# ╔═╡ 1a000000-0000-4000-8000-00000000000d
md"""
## 5. Evaluation: 5-fold cross-validation
One split can be lucky. Five folds give a mean and a spread.
"""

# ╔═╡ 1a000000-0000-4000-8000-00000000000e
begin
    dev = vcat(s.train, s.val)
    cv1 = cross_validate(fit_ols, df, dev, 5)
    (folds = round.(cv1; digits = 4), mean = mean(cv1), std = std(cv1))
end

# ╔═╡ 1a000000-0000-4000-8000-00000000000f
md"""
## 6. Improvement: one-hot categoricals + engineered features + ridge

Pick λ. Zero is plain OLS; larger shrinks the coefficients.
"""

# ╔═╡ 1a000000-0000-4000-8000-000000000010
λ = 30.0

# ╔═╡ 1a000000-0000-4000-8000-000000000011
begin
    f2 = fit_features_v2(df, s.train)
    X2_tr = design_matrix(f2, df, s.train); X2_va = design_matrix(f2, df, s.val)
    m2 = fit_ridge(X2_tr, y_tr, λ)
    (columns = size(X2_tr, 2), one_hot = length(f2.levels),
     val_rmse = rmse(y_va, predict(m2, X2_va)), val_r2 = r2(y_va, predict(m2, X2_va)))
end

# ╔═╡ 1a000000-0000-4000-8000-000000000012
md"Which houses does the model still get most wrong?"

# ╔═╡ 1a000000-0000-4000-8000-000000000013
worst_residuals(df, s.val, predict(m2, X2_va); top = 8)

# ╔═╡ 1a000000-0000-4000-8000-000000000014
md"""
## 7. Deployment: the saved model and a prediction

`scripts/05_train_final.jl` refits on train+val, scores the test set once,
and writes `model/homestead_v2.json`. `src/serve.jl` wraps it in an HTTP endpoint.
Edit the house below:
"""

# ╔═╡ 1a000000-0000-4000-8000-000000000015
house = Dict("Gr Liv Area" => 1500, "Overall Qual" => 7, "Year Built" => 2005, "Yr Sold" => 2010,
             "Neighborhood" => "CollgCr", "Total Bsmt SF" => 1000, "Garage Cars" => 2, "Full Bath" => 2)

# ╔═╡ 1a000000-0000-4000-8000-000000000016
begin
    sm = load_model(joinpath(@__DIR__, "..", "model", "homestead_v2.json"))
    p = predict_price(sm, house)
    (predicted_price = round(Int, p.price), test_rmse_log = sm.metrics["test_rmse_log"], test_r2 = sm.metrics["test_r2"])
end

# ╔═╡ Cell order:
# ╠═1a000000-0000-4000-8000-000000000001
# ╟─1a000000-0000-4000-8000-000000000002
# ╟─1a000000-0000-4000-8000-000000000003
# ╠═1a000000-0000-4000-8000-000000000004
# ╟─1a000000-0000-4000-8000-000000000005
# ╠═1a000000-0000-4000-8000-000000000006
# ╟─1a000000-0000-4000-8000-000000000007
# ╠═1a000000-0000-4000-8000-000000000008
# ╟─1a000000-0000-4000-8000-000000000009
# ╠═1a000000-0000-4000-8000-00000000000a
# ╟─1a000000-0000-4000-8000-00000000000b
# ╠═1a000000-0000-4000-8000-00000000000c
# ╟─1a000000-0000-4000-8000-00000000000d
# ╠═1a000000-0000-4000-8000-00000000000e
# ╟─1a000000-0000-4000-8000-00000000000f
# ╠═1a000000-0000-4000-8000-000000000010
# ╠═1a000000-0000-4000-8000-000000000011
# ╟─1a000000-0000-4000-8000-000000000012
# ╠═1a000000-0000-4000-8000-000000000013
# ╟─1a000000-0000-4000-8000-000000000014
# ╠═1a000000-0000-4000-8000-000000000015
# ╠═1a000000-0000-4000-8000-000000000016
