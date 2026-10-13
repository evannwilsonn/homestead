"""
Month-by-month replay of the Ames sales, 2006-01 to 2010-07.

Each month, in this order, so nothing sees the future:
  1. every system predicts that month's houses BEFORE learning their prices
     (prequential evaluation: the score is always on unseen sales)
  2. the systems absorb the month's sales
  3. the self-improving system runs its recipe search on what it has seen,
     and promotes a new recipe only if it beats the current one on the most
     recent months (held out from that comparison's training)
  4. a drift monitor compares the recent *median* absolute error with its
     long-run level and, when it jumps, gives the next search more generations.
     Median, not RMSE: one \$13,100 foreclosure in a 28-sale month (Feb 2008)
     quadruples that month's RMSE but says nothing about the market.

Three systems run side by side from the same starting recipe:
  frozen   fit once on the warm-up months, never touched again
  online   same recipe, weights updated with every sale (RLS)
  system   online updates + monthly recipe evolution + drift response
"""

using DataFrames, Statistics

struct MonthResult
    month::String
    n::Int
    rmse_frozen::Float64
    rmse_online::Float64
    rmse_system::Float64
    medae_system::Float64
    recipe_features::Int
    λ::Float64
    promoted::Bool
    drift::Bool
    generations::Int
end

mutable struct Deployed
    recipe::Recipe
    t::Transform
    model::OnlineRidge
end

function deploy(bank, y, r::Recipe, rows)
    t = fit_transform(bank, r, rows)
    Deployed(r, t, OnlineRidge(build(bank, t, rows), y[rows], 10^r.logλ))
end
predict(d::Deployed, bank, rows) = predict(d.model, build(bank, d.t, rows))
absorb!(d::Deployed, bank, y, rows) = update!(d.model, build(bank, d.t, rows), y[rows])

monthkey(df) = [string(y, "-", lpad(m, 2, '0')) for (y, m) in zip(df[!, "Yr Sold"], df[!, "Mo Sold"])]

"""
replay(df; warmup = 6, valid_months = 3) → (results, promotions)

`df` must be the cleaned table with engineered inputs available (clean(load_raw())).
"""
function replay(df::DataFrame; warmup = 6, valid_months = 3, seed_recipe = nothing,
                base_gens = 3, drift_gens = 8, drift_ratio = 1.25, min_gain = 0.05, verbose = true)
    bank = FeatureBank(df)
    y = Vector{Float64}(df[!, "LogSalePrice"])
    keys_ = monthkey(df)
    months = sort(unique(keys_))
    rows_of = Dict(m => findall(==(m), keys_) for m in months)

    seen = reduce(vcat, [rows_of[m] for m in months[1:warmup]])
    r0 = something(seed_recipe, Recipe(bank, numeric_features(df), String[], 1.0))
    frozen = deploy(bank, y, r0, seen)
    online = deploy(bank, y, r0, seen)
    system = deploy(bank, y, r0, seen)
    pop = Population(r0)

    results, promotions = MonthResult[], NamedTuple[]
    gens = base_gens
    for (mi, m) in enumerate(months[warmup+1:end])
        rows = rows_of[m]
        yt = y[rows]
        e_frozen = rmse(yt, predict(frozen, bank, rows))
        e_online = rmse(yt, predict(online, bank, rows))
        ŷs = predict(system, bank, rows)
        e_system = rmse(yt, ŷs)
        medae = median(abs.(yt .- ŷs))

        absorb!(online, bank, y, rows)
        absorb!(system, bank, y, rows)
        seen = vcat(seen, rows)

        # recipe search: train on seen months except the most recent ones, validate on those
        seen_months = months[1:warmup+mi]
        valid = reduce(vcat, [rows_of[k] for k in seen_months[end-valid_months+1:end]])
        train = setdiff(seen, valid)
        best, best_fit = evolve!(pop, bank, y, train, valid; generations = gens)
        champ_fit = fitness(bank, y, system.recipe, train, valid)
        promoted = best != system.recipe && best_fit < champ_fit * (1 - min_gain)
        if promoted
            system = deploy(bank, y, best, seen)
            push!(promotions, (month = m, from_fit = champ_fit, to_fit = best_fit, recipe = describe_recipe(bank, best)))
        end

        # drift: recent typical error vs its long-run level
        hist = [r.medae_system for r in results]
        recent = vcat(hist, medae)[max(1, end-2):end]
        drift = length(hist) >= 6 && mean(recent) > drift_ratio * median(hist)
        push!(results, MonthResult(m, length(rows), e_frozen, e_online, e_system, medae,
                                   nfeatures(system.recipe), 10^system.recipe.logλ, promoted, drift, gens))
        gens = drift ? drift_gens : base_gens
        verbose && println(rpad(m, 8), lpad(length(rows), 4), "  frozen ", round(e_frozen, digits = 4),
                           "  online ", round(e_online, digits = 4), "  system ", round(e_system, digits = 4),
                           "  feats ", nfeatures(system.recipe), promoted ? "  PROMOTED" : "", drift ? "  DRIFT" : "")
    end
    results, promotions, bank, system
end
