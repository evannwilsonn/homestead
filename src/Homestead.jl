module Homestead

include("data.jl")
include("metrics.jl")
include("baseline.jl")
include("features.jl")
include("linear.jl")
include("evaluate.jl")
include("model_io.jl")

export load_raw, clean, split_indices
export rmse, mae, r2, pct_error
export MedianBaseline, OneFeatureBaseline, fit_median, fit_one_feature, predict
export numeric_features, Standardizer, fit_standardizer, design_matrix
export add_features, FeaturesV2, fit_features_v2, feature_names, CATEGORICALS
export LinearModel, fit_ols, fit_ols_normal, fit_ridge, fit_gd
export kfold, cross_validate, worst_residuals
export SavedModel, save_model, load_model, predict_price

end # module
