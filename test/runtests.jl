using Test, DataFrames, Statistics, Dates
include(joinpath(@__DIR__, "..", "src", "Homestead.jl"))
using .Homestead

@testset "data loading" begin
    raw = load_raw()
    @test nrow(raw) == 2930
    @test ncol(raw) == 82
    @test "SalePrice" in names(raw)

    df = clean(raw)
    @test nrow(df) == 2925            # 5 De Cock outliers removed
    @test !("PID" in names(df))
    @test all(!ismissing, Matrix(df))  # no missings left
    @test all(df[!, "LogSalePrice"] .≈ log.(df[!, "SalePrice"]))

    s = split_indices(nrow(df))
    @test length(s.train) + length(s.val) + length(s.test) == nrow(df)
    @test isempty(intersect(s.train, s.test))
end

@testset "baselines" begin
    y = [1.0, 2.0, 3.0, 10.0]
    @test predict(fit_median(y), 2) == [2.5, 2.5]
    x = [1.0, 2.0, 3.0, 4.0]; yl = 2 .* x .+ 1          # exact line, zero residual
    m = fit_one_feature(x, yl)
    @test m.slope ≈ 2.0 && m.intercept ≈ 1.0
    @test rmse(yl, predict(m, x)) ≈ 0.0 atol = 1e-12
    @test r2(yl, predict(m, x)) ≈ 1.0
    @test pct_error(log(1.1)) ≈ 10.0
end

@testset "linear models from scratch" begin
    using LinearAlgebra, Random
    rng = MersenneTwister(1)
    X = hcat(ones(200), randn(rng, 200, 3))
    β_true = [3.0, 1.5, -2.0, 0.5]
    y = X * β_true .+ 0.01 .* randn(rng, 200)

    @test fit_ols(X, y).β ≈ β_true atol = 1e-2
    @test fit_ols_normal(X, y).β ≈ fit_ols(X, y).β atol = 1e-10
    @test fit_ridge(X, y, 0.0).β ≈ fit_ols(X, y).β atol = 1e-10
    @test norm(fit_ridge(X, y, 100.0).β[2:end]) < norm(fit_ols(X, y).β[2:end])   # shrinkage
    gd, hist = fit_gd(X, y; lr = 0.1, epochs = 1000)
    @test gd.β ≈ fit_ols(X, y).β atol = 1e-4
    @test all(diff(hist) .<= 1e-12)                                              # loss never rises
end

@testset "design matrix" begin
    df = clean(load_raw()); s = split_indices(nrow(df))
    cols = numeric_features(df); st = fit_standardizer(df, s.train, cols)
    Xtr = design_matrix(st, df, s.train)
    @test all(Xtr[:, 1] .== 1.0)
    @test maximum(abs.(mean(Xtr[:, 2:end], dims = 1))) < 1e-10        # standardized on train
    @test rank(Xtr' * Xtr) == size(Xtr, 2)                             # invertible: no exact sums left
end

@testset "cross-validation" begin
    folds = kfold(103, 5)
    @test length(folds) == 5
    @test sort(vcat(folds...)) == 1:103                    # every row in exactly one fold
    @test maximum(length.(folds)) - minimum(length.(folds)) <= 1
    df = clean(load_raw()); s = split_indices(nrow(df))
    cv = cross_validate(fit_ols, df, s.train, 3)
    @test length(cv) == 3 && all(0.05 .< cv .< 0.3)
end

@testset "features v2" begin
    df = add_features(clean(load_raw())); s = split_indices(nrow(df))
    @test all(df[!, "Age"] .>= -1)                                     # sold the year before built happens once in Ames
    f = fit_features_v2(df, s.train)
    X = design_matrix(f, df, s.val)
    @test size(X, 2) == length(feature_names(f))
    @test all(x -> x == 0.0 || x == 1.0, X[:, end-length(f.levels)+1:end])   # dummies are 0/1
    @test !any(l -> l == ("Neighborhood", "NAmes"), f.levels)         # most common level is the reference
end

@testset "online ridge (RLS) matches batch ridge" begin
    using Random, LinearAlgebra
    rng = MersenneTwister(3)
    X = hcat(ones(300), randn(rng, 300, 6)); y = X * randn(rng, 7) .+ 0.1 .* randn(rng, 300)
    m = OnlineRidge(X[1:50, :], y[1:50], 5.0)
    update!(m, X[51:300, :], y[51:300])
    @test m.β ≈ fit_ridge(X, y, 5.0).β atol = 1e-8         # absorbing 250 sales one by one == refitting on all 300
    @test m.n == 300
end

@testset "recipes and genetic search" begin
    using Random
    df = clean(load_raw()); bank = FeatureBank(df)
    @test !("Yr Sold" in bank.names) && !("SalePrice" in bank.names)       # time and target never become features
    r = Recipe(bank, ["Gr Liv Area", "Overall Qual"], ["Neighborhood"], 10.0)
    @test nfeatures(r) == 3
    t = fit_transform(bank, r, 1:1000)
    X = build(bank, t, 1001:1100)
    @test size(X, 2) == 1 + 2 + length(t.dummies) && all(X[:, 1] .== 1)
    @test all(x -> x == 0 || x == 1, X[:, 4:end])
    y = Vector{Float64}(df[!, "LogSalePrice"])
    p1 = Population(r; rng = MersenneTwister(1)); b1, f1 = evolve!(p1, bank, y, 1:1500, 1501:2000; generations = 2)
    p2 = Population(r; rng = MersenneTwister(1)); b2, f2 = evolve!(p2, bank, y, 1:1500, 1501:2000; generations = 2)
    @test b1 == b2 && f1 == f2                                              # same seed, same search
    @test f1 <= fitness(bank, y, r, 1:1500, 1501:2000)                      # elitism: never worse than the seed
end

@testset "decade exam inputs" begin
    hpi = load_hpi()
    @test hpi[(2006, 4)] ≈ 164.36 && hpi[(2022, 4)] ≈ 269.20                     # Ames, IA metro (CBSA 11180)
    @test Homestead.lagged(hpi, 2021, 1) == hpi[(2020, 4)]                        # a 2021-Q1 sale only knows 2020-Q4
    @test Homestead.lagged(hpi, 2021, 3) == hpi[(2021, 2)]
    @test Homestead.parse_area("1,295, 1,295") === nothing && Homestead.parse_area(2062) == 2062
    p = repeat_pairs()
    @test nrow(p) == 422
    @test all(abs.(p.area .- p[!, "Gr Liv Area"]) .<= 50)
    @test all(year.(p.date) .>= 2020) && all(p[!, "Yr Sold"] .<= 2010)           # every pair goes forward in time
end
