# Fit the drift correction on all 2020–2022 repeat sales and export what the
# browser app needs to show today's-dollar estimates.
#   julia --project=. scripts/08_export_today.jl
include(joinpath(@__DIR__, "..", "src", "Homestead.jl"))
using .Homestead, DataFrames, Dates, Statistics, JSON3

const λ = 10.0                      # same as the decade exam
p = repeat_pairs(); hpi = load_hpi()
sm = load_model(joinpath(@__DIR__, "..", "model", "homestead_v2.json"))
raw_cols = [c for c in names(p) if haskey(sm.defaults, c)]
q(m) = (m - 1) ÷ 3 + 1
y = log.(Float64.(p[!, "Sale Amount"]))
f = [predict_price(sm, Dict{String,Any}(c => p[i, c] for c in raw_cols)).log_price for i in 1:nrow(p)]
idx = [log(Homestead.lagged(hpi, year(d), q(month(d))) / hpi[(p[i, "Yr Sold"], q(p[i, "Mo Sold"]))]) for (i, d) in enumerate(p.date)]
years = [Dates.value(d - Date(p[i, "Yr Sold"], p[i, "Mo Sold"], 15)) / 365.25 for (i, d) in enumerate(p.date)]
nbs = sort(unique(p.Neighborhood)); ref = nbs[1]; nbs = nbs[2:end]
Z = reduce(hcat, [vcat(1.0, [p.Neighborhood[i] == n ? 1.0 : 0.0 for n in nbs], f[i] - 12.0, (years[i] - 13.0) / 2) for i in 1:nrow(p)])' |> Matrix
m = OnlineRidge(Z, y .- f .- idx, λ)
resid = y .- (f .+ idx .+ Z * m.β)
println("in-sample RMSE (all 422, for reference only): ", round(sqrt(mean(resid .^ 2)), digits = 4))

out = Dict(
    "reference_neighborhood" => ref, "neighborhoods" => nbs, "beta" => m.β,
    "hpi" => Dict(string(y, "Q", qq) => hpi[(y, qq)] for y in 2006:2010 for qq in 1:4),
    "target" => Dict("label" => "late 2022", "index" => Homestead.lagged(hpi, 2022, 4)),
    "latest" => Dict("label" => "mid-2026", "index" => hpi[(2026, 2)], "quarter" => "2026Q2"),
    "exam" => Dict("rmse_log" => 0.1392, "n" => 394, "mae_usd" => 25967),
    "target_date" => "2022-11-15", "n_pairs" => nrow(p))
open(joinpath(@__DIR__, "..", "model", "today_correction.json"), "w") do io; JSON3.pretty(io, out); end
# one worked example for the browser port to match
ex = Dict{String,Any}("Gr Liv Area" => 1500, "Overall Qual" => 7, "Year Built" => 2005, "Neighborhood" => "CollgCr", "Yr Sold" => 2008, "Mo Sold" => 5)
fe = predict_price(sm, ex).log_price
ye = (Dates.value(Date(2022, 11, 15) - Date(2008, 5, 15)) / 365.25)
ze = vcat(1.0, [n == "CollgCr" ? 1.0 : 0.0 for n in nbs], fe - 12.0, (ye - 13.0) / 2)
println("example today log price: ", fe + log(Homestead.lagged(hpi, 2022, 4) / hpi[(2008, 2)]) + ze' * m.β)
println("neighborhood offsets (vs ", ref, "): ", join([string(n, " ", round(b, digits = 3)) for (n, b) in zip(nbs, m.β[2:end-2])], ", "))
println("tier coef ", round(m.β[end-1], digits = 3), "  years coef ", round(m.β[end], digits = 3), "  intercept ", round(m.β[1], digits = 3))
