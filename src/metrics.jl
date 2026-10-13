"""
Evaluation metrics. Everything is computed on log(SalePrice), so an RMSE of
0.10 means predictions are typically off by about 10% of the price.
"""

using Statistics

"Root mean squared error."
rmse(y, ŷ) = sqrt(mean((y .- ŷ) .^ 2))

"Mean absolute error."
mae(y, ŷ) = mean(abs.(y .- ŷ))

"Coefficient of determination: 1 minus (residual variance / total variance)."
r2(y, ŷ) = 1 - sum((y .- ŷ) .^ 2) / sum((y .- mean(y)) .^ 2)

"Turn a log-scale RMSE into the typical percentage error in dollars."
pct_error(rmse_log) = (exp(rmse_log) - 1) * 100
