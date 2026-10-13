# Decade-later exam on 2020–2022 repeat sales.
#   julia --project=. scripts/07_decade.jl
include(joinpath(@__DIR__, "..", "src", "Homestead.jl"))
using .Homestead, DataFrames, Printf, Statistics, CSV

res, preds = decade_exam()
n = sum(r.n for r in res)
names_ = [(:frozen, "Frozen 2010 model"), (:frozen_index, "Frozen + FHFA index"), (:repeat_sales, "Repeat sales (old price × index)"),
          (:system, "Self-improving correction"), (:system_history, "Self-improving + house's own history")]
println("Scored sales: ", n, " (", res[1].month, " to ", res[end].month, "; first 3 months seed the corrections)")
for (k, label) in names_
    rm = sqrt(sum(r.sq[k] for r in res) / n); mae = sum(r.ae[k] for r in res) / n
    byyear = [sqrt(sum(r.sq[k] for r in res if startswith(r.month, y)) / sum(r.n for r in res if startswith(r.month, y))) for y in ("2020", "2021", "2022")]
    @printf "%-38s RMSE %.4f (≈ ±%4.1f%%)  MAE \$%6.0f   by year %s\n" label rm pct_error(rm) mae join([@sprintf("%.3f", b) for b in byyear], " / ")
end
mkpath(joinpath(@__DIR__, "..", "results"))
CSV.write(joinpath(@__DIR__, "..", "results", "decade_predictions.csv"), preds)
