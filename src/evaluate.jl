"""
Evaluation tools.

`kfold` splits row indices into k folds. `cross_validate` fits a model on k−1
folds and scores it on the held-out fold, k times, and returns every fold's
RMSE. The mean is the estimate; the spread tells you how much to trust it.
A single validation split can get lucky or unlucky; k folds average that out.

`worst_residuals` ranks houses by how far the model missed, in log space, so
you can look at what the model does not understand.
"""

using DataFrames, Random, Statistics

"Split 1:n into k roughly equal folds, shuffled with a fixed seed."
function kfold(n::Integer, k::Integer; seed = 42)
    idx = shuffle(MersenneTwister(seed), 1:n)
    [idx[i:k:n] for i in 1:k]
end

"""
cross_validate(fit, df, rows, k) → Vector of k RMSEs

`fit(X, y)` must return something `predict(m, X)` accepts. `featurizer(df, rows)`
builds the feature transform; it is refitted inside every fold on that fold's training rows only, so no
information from the held-out fold leaks into the features.
"""
function cross_validate(fit, df::DataFrame, rows, k::Integer; seed = 42,
                        featurizer = (df, r) -> fit_standardizer(df, r, numeric_features(df)))
    folds = kfold(length(rows), k; seed)
    scores = Float64[]
    for i in 1:k
        test_rows  = rows[folds[i]]
        train_rows = rows[vcat(folds[1:k .!= i]...)]
        st = featurizer(df, train_rows)
        m  = fit(design_matrix(st, df, train_rows), df[train_rows, "LogSalePrice"])
        push!(scores, rmse(df[test_rows, "LogSalePrice"], predict(m, design_matrix(st, df, test_rows))))
    end
    scores
end

"Houses ordered by |actual − predicted| in log price, largest first."
function worst_residuals(df::DataFrame, rows, ŷ; top = 10)
    actual = df[rows, "SalePrice"]
    pred   = exp.(ŷ)
    out = DataFrame(
        row = rows,
        actual = actual,
        predicted = round.(Int, pred),
        resid_log = log.(actual) .- ŷ,
        neighborhood = df[rows, "Neighborhood"],
        sale_cond = df[rows, "Sale Condition"],
        living_sf = df[rows, "Gr Liv Area"],
        year_built = df[rows, "Year Built"],
    )
    sort!(out, :resid_log, by = abs, rev = true)
    first(out, top)
end
