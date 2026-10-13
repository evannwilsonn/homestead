"""
Baselines. Two deliberately dumb models whose scores every real model has to beat.

1. `MedianBaseline`  — predict the training-set median log-price for every house.
2. `OneFeatureBaseline` — a straight line through (Gr Liv Area, log price),
   fitted with the two-variable least-squares formula:
       slope     = cov(x, y) / var(x)
       intercept = mean(y) - slope * mean(x)
"""

using Statistics

struct MedianBaseline
    value::Float64
end
fit_median(y) = MedianBaseline(median(y))
predict(m::MedianBaseline, n::Integer) = fill(m.value, n)

struct OneFeatureBaseline
    slope::Float64
    intercept::Float64
end
function fit_one_feature(x, y)
    slope = cov(x, y) / var(x)
    OneFeatureBaseline(slope, mean(y) - slope * mean(x))
end
predict(m::OneFeatureBaseline, x) = m.intercept .+ m.slope .* x
