"""
The decade-later exam.

The City of Ames 2020–2022 sales include houses that also sold in 2006–2010
(same parcel). For each such repeat sale whose living area is unchanged, every
method predicts the 2020–2022 price using only what was known before that sale:

  frozen         the 2010 ridge model, as if nothing changed since 2010
  frozen+index   that prediction scaled by the FHFA Ames house price index
  repeat-sales   the house's own earlier price scaled by the same index
                 (the textbook appraisal shortcut)
  system         frozen+index plus a correction learned online, month by
                 month, from the 2020–2022 repeat sales as they arrive:
                 neighborhood drift, price tier, years elapsed
  system+history the same, also told how far the house's earlier price sat
                 from what the 2010 model expected (its unmeasured quirks)

The index is lagged one quarter: FHFA publishes a couple of months late, so a
sale in a given quarter can only use the previous quarter's index.
"""

using CSV, DataFrames, Dates, Statistics, XLSX

const NEW_SALES_PATH = joinpath(@__DIR__, "..", "data", "raw", "ames_residential_sales_2020_2022.xlsx")
const HPI_PATH = joinpath(@__DIR__, "..", "data", "raw", "fhfa_hpi_metro.csv")

"FHFA all-transactions index for the Ames, IA metro area, keyed by (year, quarter)."
function load_hpi(path = HPI_PATH)
    hpi = Dict{Tuple{Int,Int},Float64}()
    for row in CSV.File(path; header = [:name, :cbsa, :year, :qtr, :index, :se], missingstring = "-")
        row.cbsa == 11180 && !ismissing(row.index) && (hpi[(row.year, row.qtr)] = row.index)
    end
    hpi
end
quarter(m) = (m - 1) ÷ 3 + 1
lagged(hpi, y, q) = q == 1 ? hpi[(y - 1, 4)] : hpi[(y, q - 1)]

"Parse an Area cell; multi-building parcels like \"1,295, 1,295\" return nothing."
parse_area(v) = (s = replace(string(v), "," => ""); all(isdigit, s) && !isempty(s) ? parse(Int, s) : nothing)

"""
repeat_pairs() → DataFrame, one row per 2020–2022 sale of a parcel that also
sold in 2006–2010 with the same living area (±50 sq ft). The old side is the
parcel's most recent 2006–2010 sale, with every raw column kept.
"""
function repeat_pairs()
    raw = load_raw()
    keep = raw[!, "Gr Liv Area"] .<= 4000
    old = clean(raw); old[!, "PID"] = lpad.(string.(raw[keep, "PID"]), 10, '0')
    old[!, "ym"] = old[!, "Yr Sold"] .* 100 .+ old[!, "Mo Sold"]
    latest = combine(groupby(sort(old, "ym"), "PID"), last)

    new = DataFrame(XLSX.readtable(NEW_SALES_PATH, "Residential Sales"))
    new[!, "PID"] = lpad.(replace.(string.(new[!, "Parcel"]), "-" => ""), 10, '0')
    new[!, "area"] = parse_area.(new[!, "Area"])
    pairs = innerjoin(new[:, ["PID", "Sale Date", "Sale Amount", "area"]], latest; on = "PID")
    pairs = pairs[.!isnothing.(pairs.area) .& (abs.(something.(pairs.area, -9999) .- pairs[!, "Gr Liv Area"]) .<= 50), :]
    pairs[!, "date"] = Date.(pairs[!, "Sale Date"])
    sort!(pairs, "date")
end

struct DecadeResult
    month::String
    n::Int
    sq::Dict{Symbol,Float64}     # summed squared log errors per method
    ae::Dict{Symbol,Float64}     # summed absolute dollar errors per method
end

"""
decade_exam(; warmup = 3, λ = 10.0) → (monthly results, per-sale predictions)

Months are replayed in order. In each month every method predicts that
month's repeat sales first; then the online corrections absorb them. The
first `warmup` months only seed the corrections and are not scored.
"""
function decade_exam(; warmup = 3, λ = 10.0, model_path = joinpath(@__DIR__, "..", "model", "homestead_v2.json"))
    p = repeat_pairs(); hpi = load_hpi(); sm = load_model(model_path)
    raw_cols = [c for c in names(p) if haskey(sm.defaults, c)]

    y = log.(Float64.(p[!, "Sale Amount"]))
    f = [predict_price(sm, Dict{String,Any}(c => p[i, c] for c in raw_cols)).log_price for i in 1:nrow(p)]
    idx = [log(lagged(hpi, year(d), quarter(month(d))) / hpi[(p[i, "Yr Sold"], quarter(p[i, "Mo Sold"]))]) for (i, d) in enumerate(p.date)]
    old_lp = log.(Float64.(p[!, "SalePrice"]))
    years = [(Dates.value(d - Date(p[i, "Yr Sold"], p[i, "Mo Sold"], 15)) / 365.25) for (i, d) in enumerate(p.date)]

    pred = Dict(:frozen => f, :frozen_index => f .+ idx, :repeat_sales => old_lp .+ idx)

    # correction features (fixed vocabulary: neighborhoods present among the pairs)
    nbs = sort(unique(p.Neighborhood)); nbs = nbs[2:end]
    z(i, hist) = vcat(1.0, [p.Neighborhood[i] == n ? 1.0 : 0.0 for n in nbs],
                      (f[i] - 12.0), (years[i] - 13.0) / 2,
                      hist ? [old_lp[i] - f[i]] : Float64[])
    mk = Dict(:system => false, :system_history => true)
    Z = Dict(k => reduce(hcat, [z(i, h) for i in 1:nrow(p)])' |> Matrix for (k, h) in mk)
    base = pred[:frozen_index]

    months = string.(year.(p.date), "-", lpad.(month.(p.date), 2, '0'))
    order = unique(months)
    seed = findall(in(order[1:warmup]), months)
    models = Dict(k => OnlineRidge(Z[k][seed, :], y[seed] .- base[seed], λ) for k in keys(mk))
    for k in keys(mk); pred[k] = fill(NaN, nrow(p)); end

    results = DecadeResult[]
    for m in order[warmup+1:end]
        rows = findall(==(m), months)
        for k in keys(mk)
            pred[k][rows] .= base[rows] .+ predict(models[k], Z[k][rows, :])   # predict first ...
        end
        sq = Dict(k => sum(abs2, y[rows] .- v[rows]) for (k, v) in pred)
        ae = Dict(k => sum(abs.(exp.(y[rows]) .- exp.(v[rows]))) for (k, v) in pred)
        push!(results, DecadeResult(m, length(rows), sq, ae))
        for k in keys(mk)
            update!(models[k], Z[k][rows, :], y[rows] .- base[rows])            # ... then learn
        end
    end
    results, DataFrame(:date => p.date, :pid => p.PID, :neighborhood => p.Neighborhood,
                       :old_price => p.SalePrice, :new_price => p[!, "Sale Amount"],
                       [k => exp.(v) for (k, v) in pred]...)
end
