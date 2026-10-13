"""
A *recipe* is everything that decides how a house becomes a row of numbers:
which raw numeric columns, which engineered columns, which categorical columns
get one-hot encoded, and the ridge penalty λ. The self-improving system
searches over recipes; the weights inside a recipe are learned by ridge.

To keep search fast, every candidate column is computed once for the whole
table (`FeatureBank`). Building a design matrix for a recipe is then just
column selection plus standardization learned from the training rows only.
"""

using DataFrames, Statistics

"Engineered candidates. Each is computed row by row, so no information crosses between houses."
const ENGINEERED = Dict{String,Function}(
    "Age"        => df -> Float64.(df[!, "Yr Sold"] .- df[!, "Year Built"]),
    "Age2"       => df -> Float64.(df[!, "Yr Sold"] .- df[!, "Year Built"]) .^ 2,
    "RemodAge"   => df -> Float64.(df[!, "Yr Sold"] .- df[!, "Year Remod/Add"]),
    "LogLotArea" => df -> log.(Float64.(df[!, "Lot Area"])),
    "LogTotalSF" => df -> log.(Float64.(df[!, "Total Bsmt SF"] .+ df[!, "Gr Liv Area"])),
    "LogGrLiv"   => df -> log.(Float64.(df[!, "Gr Liv Area"])),
    "QualxLiv"   => df -> Float64.(df[!, "Overall Qual"] .* df[!, "Gr Liv Area"]) ./ 1000,
    "Qual2"      => df -> Float64.(df[!, "Overall Qual"]) .^ 2,
    "QualxAge"   => df -> Float64.(df[!, "Overall Qual"] .* (df[!, "Yr Sold"] .- df[!, "Year Built"])),
    "TotalBath"  => df -> Float64.(df[!, "Full Bath"] .+ 0.5 .* df[!, "Half Bath"] .+ df[!, "Bsmt Full Bath"] .+ 0.5 .* df[!, "Bsmt Half Bath"]),
    "PorchSF"    => df -> Float64.(df[!, "Open Porch SF"] .+ df[!, "Enclosed Porch"] .+ df[!, "3Ssn Porch"] .+ df[!, "Screen Porch"] .+ df[!, "Wood Deck SF"]),
    "HasGarage"  => df -> Float64.(df[!, "Garage Cars"] .> 0),
    "Has2nd"     => df -> Float64.(df[!, "2nd Flr SF"] .> 0),
)

struct FeatureBank
    names::Vector{String}            # numeric + engineered column names, in N's column order
    N::Matrix{Float64}               # n × length(names)
    cats::Vector{String}             # categorical column names
    codes::Vector{Vector{Int}}       # per categorical: integer code per row
    levels::Vector{Vector{String}}   # per categorical: level names (code → name)
end

const NOT_FEATURES = ["SalePrice", "LogSalePrice", "Yr Sold", "Mo Sold"]   # time is handled by the replay, not the recipe

function FeatureBank(df::DataFrame)
    num = [c for c in names(df) if eltype(df[!, c]) <: Number && !(c in NOT_FEATURES)]
    eng = sort(collect(keys(ENGINEERED)))
    N = hcat(Matrix{Float64}(df[!, num]), reduce(hcat, [ENGINEERED[e](df) for e in eng]))
    cats = [c for c in names(df) if eltype(df[!, c]) <: AbstractString]
    codes, levels = Vector{Int}[], Vector{String}[]
    for c in cats
        lv = sort(unique(df[!, c])); idx = Dict(l => i for (i, l) in enumerate(lv))
        push!(levels, lv); push!(codes, [idx[v] for v in df[!, c]])
    end
    FeatureBank(vcat(num, eng), N, cats, codes, levels)
end

struct Recipe
    use_num::BitVector    # over bank.names
    use_cat::BitVector    # over bank.cats
    logλ::Float64         # log10 of the ridge penalty
end
Recipe(bank::FeatureBank, num::Vector{String}, cat::Vector{String}, λ) =
    Recipe(BitVector(n in num for n in bank.names), BitVector(c in cat for c in bank.cats), log10(λ))

nfeatures(r::Recipe) = count(r.use_num) + count(r.use_cat)
Base.:(==)(a::Recipe, b::Recipe) = a.use_num == b.use_num && a.use_cat == b.use_cat && a.logλ == b.logλ

describe_recipe(bank, r::Recipe) = (numeric = bank.names[r.use_num], categorical = bank.cats[r.use_cat], λ = round(10^r.logλ, sigdigits = 3))

"""
The transform a recipe learns from training rows: standardization stats for its
numeric columns and the one-hot levels it keeps (seen ≥ `min_count` times; the
most common level is dropped as the reference).
"""
struct Transform
    num_idx::Vector{Int}
    μ::Vector{Float64}
    σ::Vector{Float64}
    dummies::Vector{Tuple{Int,Int}}   # (categorical index, level code)
end

function fit_transform(bank::FeatureBank, r::Recipe, rows; min_count = 10)
    num_idx = findall(r.use_num)
    M = bank.N[rows, num_idx]
    μ = vec(mean(M, dims = 1)); σ = vec(std(M, dims = 1)); σ[σ .< 1e-12] .= 1.0
    dummies = Tuple{Int,Int}[]
    for ci in findall(r.use_cat)
        counts = zeros(Int, length(bank.levels[ci]))
        for k in bank.codes[ci][rows]; counts[k] += 1; end
        keep = [k for k in sortperm(counts, rev = true) if counts[k] >= min_count]
        append!(dummies, (ci, k) for k in keep[2:end])
    end
    Transform(num_idx, μ, σ, dummies)
end

"Design matrix: intercept, standardized numerics, then 0/1 dummies."
function build(bank::FeatureBank, t::Transform, rows)
    X = Matrix{Float64}(undef, length(rows), 1 + length(t.num_idx) + length(t.dummies))
    X[:, 1] .= 1.0
    X[:, 2:1+length(t.num_idx)] .= (bank.N[rows, t.num_idx] .- t.μ') ./ t.σ'
    off = 1 + length(t.num_idx)
    for (j, (ci, k)) in enumerate(t.dummies)
        X[:, off+j] .= bank.codes[ci][rows] .== k
    end
    X
end
