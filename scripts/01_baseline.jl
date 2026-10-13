# Step 3 of the loop: baselines. Run from the project root:
#   julia --project=. scripts/01_baseline.jl
include(joinpath(@__DIR__, "..", "src", "Homestead.jl"))
using .Homestead, DataFrames, Printf

df = clean(load_raw())
s  = split_indices(nrow(df))
y_train = df[s.train, "LogSalePrice"];  y_val = df[s.val, "LogSalePrice"]
x_train = Float64.(df[s.train, "Gr Liv Area"]);  x_val = Float64.(df[s.val, "Gr Liv Area"])

m1 = fit_median(y_train)
r1 = rmse(y_val, predict(m1, length(y_val)))
@printf "Median baseline      RMSE(log) = %.4f  ≈ ±%.1f%%  (median price \$%.0f)\n" r1 pct_error(r1) exp(m1.value)

m2 = fit_one_feature(x_train, y_train)
ŷ2 = predict(m2, x_val)
r2v = rmse(y_val, ŷ2)
@printf "Living-area line     RMSE(log) = %.4f  ≈ ±%.1f%%  R² = %.3f  slope = %.5f /sqft\n" r2v pct_error(r2v) r2(y_val, ŷ2) m2.slope
