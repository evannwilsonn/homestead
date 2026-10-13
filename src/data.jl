"""
Data loading and cleaning for the Ames Housing dataset.

Source: De Cock, D. (2011). "Ames, Iowa: Alternative to the Boston Housing
Data as an End of Semester Regression Project." Journal of Statistics
Education 19(3). https://jse.amstat.org/v19n3/decock.pdf
Raw file: data/raw/AmesHousing.txt (tab-separated, 2,930 rows, 82 columns).
"""

using CSV, DataFrames, Statistics, Random

const RAW_PATH = joinpath(@__DIR__, "..", "data", "raw", "AmesHousing.txt")

"Read the raw tab-separated file into a DataFrame. Column names keep their spaces."
function load_raw(path::AbstractString = RAW_PATH)
    CSV.read(path, DataFrame; delim = '\t', missingstring = ["NA", ""], normalizenames = false)
end

"""
Clean the raw frame into a modelling table.

- Drops the `Order` and `PID` identifier columns.
- Removes the five partial-sale / >4000 sq ft outliers De Cock flags in the paper.
- Fills numeric missings with the column median, categorical missings with "None"
  (in Ames "NA" for e.g. Garage Type means the house has no garage, not unknown).
- Adds the log target `LogSalePrice` so errors are relative, not absolute.
"""
function clean(df::DataFrame)
    out = select(df, Not(["Order", "PID"]))
    out = out[out[!, "Gr Liv Area"] .<= 4000, :]

    for col in names(out)
        v = out[!, col]
        if eltype(v) <: Union{Missing, Number}
            med = median(skipmissing(v))
            out[!, col] = coalesce.(v, med)
        else
            out[!, col] = String.(coalesce.(v, "None"))
        end
    end
    out[!, "LogSalePrice"] = log.(out[!, "SalePrice"])
    out
end

"""
Split row indices into train / validation / test (default 70 / 15 / 15).
The test set is held out and touched exactly once, at the end.
"""
function split_indices(n::Integer; train = 0.70, val = 0.15, seed = 42)
    rng = MersenneTwister(seed)
    idx = shuffle(rng, 1:n)
    n_train = round(Int, train * n)
    n_val   = round(Int, val * n)
    (train = idx[1:n_train],
     val   = idx[n_train+1:n_train+n_val],
     test  = idx[n_train+n_val+1:end])
end
