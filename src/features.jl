"""
Turn the cleaned table into a numeric design matrix X.

Version 1 uses the numeric columns only, standardized to mean 0 / std 1 using
the TRAINING rows (never the validation or test rows, or information leaks).
A column of ones is prepended for the intercept.

Three columns are dropped because they are exact sums of other columns
(Gr Liv Area = 1st Flr SF + 2nd Flr SF + Low Qual Fin SF, and so on). Keeping
both a total and its parts makes XᵀX singular, and the normal equation
cannot invert a singular matrix.
"""

using DataFrames, Statistics

const TARGETS = ["SalePrice", "LogSalePrice"]
const DERIVED = ["Gr Liv Area", "Total Bsmt SF", "TotRms AbvGrd"]

"Names of the numeric feature columns in `df`."
function numeric_features(df::DataFrame)
    [c for c in names(df) if eltype(df[!, c]) <: Number && !(c in TARGETS) && !(c in DERIVED)]
end

struct Standardizer
    cols::Vector{String}
    μ::Vector{Float64}
    σ::Vector{Float64}
end

"Learn column means and stds from the training rows only."
function fit_standardizer(df::DataFrame, rows, cols::Vector{String})
    M = Matrix{Float64}(df[rows, cols])
    μ = vec(mean(M, dims = 1))
    σ = vec(std(M, dims = 1))
    σ[σ .== 0] .= 1.0                    # constant column: leave it alone
    Standardizer(cols, μ, σ)
end

"Build X for the given rows: intercept column, then standardized features."
function design_matrix(st::Standardizer, df::DataFrame, rows)
    M = (Matrix{Float64}(df[rows, st.cols]) .- st.μ') ./ st.σ'
    hcat(ones(size(M, 1)), M)
end

# ---------------------------------------------------------------------------
# Version 2: engineered columns + one-hot categoricals.
#
# Error analysis (scripts/03_evaluate.jl) showed the worst misses were cheap
# old houses in a handful of neighborhoods. Neighborhood is a string column
# the v1 matrix never saw. One-hot encoding turns each category level into a
# 0/1 column so a linear model can give every neighborhood its own offset.
#
# Engineered columns are chosen to be NON-linear in the existing columns
# (logs, squares, products): an exact linear combination of columns already
# in X would make XᵀX singular again.
# ---------------------------------------------------------------------------

"Add engineered columns. Returns a new DataFrame."
function add_features(df::DataFrame)
    out = copy(df)
    out[!, "Age"]        = Float64.(out[!, "Yr Sold"] .- out[!, "Year Built"])
    out[!, "Age2"]       = out[!, "Age"] .^ 2
    out[!, "RemodAge"]   = Float64.(out[!, "Yr Sold"] .- out[!, "Year Remod/Add"])
    out[!, "LogLotArea"] = log.(Float64.(out[!, "Lot Area"]))
    out[!, "LogTotalSF"] = log.(Float64.(out[!, "Total Bsmt SF"] .+ out[!, "Gr Liv Area"]))
    out[!, "QualxLiv"]   = Float64.(out[!, "Overall Qual"] .* out[!, "Gr Liv Area"]) ./ 1000
    out
end

"String columns worth encoding. Levels seen fewer than `min_count` times in training are lumped into the dropped reference level."
const CATEGORICALS = ["Neighborhood", "MS Zoning", "Bldg Type", "House Style", "Exter Qual",
                      "Kitchen Qual", "Bsmt Qual", "Garage Finish", "Sale Condition", "Foundation",
                      "Central Air", "Heating QC", "Fireplace Qu", "Garage Type", "Lot Shape"]

struct FeaturesV2
    numeric::Standardizer
    levels::Vector{Tuple{String,String}}   # (column, level) → one dummy column each
end

"Learn standardization and one-hot levels from the training rows only."
function fit_features_v2(df::DataFrame, rows; min_count = 10)
    st = fit_standardizer(df, rows, numeric_features(df))
    levels = Tuple{String,String}[]
    for col in CATEGORICALS
        counts = Dict{String,Int}()
        for v in df[rows, col]; counts[v] = get(counts, v, 0) + 1; end
        kept = sort([l for (l, c) in counts if c >= min_count], by = l -> -counts[l])
        append!(levels, (col, l) for l in kept[2:end])   # drop the most common level as reference
    end
    FeaturesV2(st, levels)
end

"Intercept, standardized numerics, then one 0/1 column per kept level."
function design_matrix(f::FeaturesV2, df::DataFrame, rows)
    X = design_matrix(f.numeric, df, rows)
    D = zeros(length(rows), length(f.levels))
    for (j, (col, lvl)) in enumerate(f.levels)
        D[:, j] .= df[rows, col] .== lvl
    end
    hcat(X, D)
end

"Column names matching `design_matrix(f, ...)`, for reading coefficients."
feature_names(f::FeaturesV2) = vcat("intercept", f.numeric.cols, ["$c=$l" for (c, l) in f.levels])
