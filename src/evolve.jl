"""
Genetic search over recipes. This keeps the evolutionary idea from NEAT but
evolves what is worth searching on tabular data (which columns, which
engineered features, which categories, how much regularization) and leaves
the weights to ridge, which finds them exactly.

Genome: one bit per candidate column plus log10(λ).
Fitness: RMSE on a validation window of recent months, after training on the
months before it. A tiny per-feature cost breaks ties toward simpler recipes.
The population persists from month to month, so the search keeps what it has
learned instead of restarting.
"""

using Random, Statistics

mutable struct Population
    members::Vector{Recipe}
    rng::MersenneTwister
end

const FEATURE_COST = 2e-5   # RMSE units per feature; only decides near-ties

"Score a recipe: fit on `train`, RMSE on `valid` (plus the parsimony cost)."
function fitness(bank::FeatureBank, y::Vector{Float64}, r::Recipe, train, valid)
    t = fit_transform(bank, r, train)
    Xt = build(bank, t, train)
    p = size(Xt, 2)
    D = Matrix{Float64}(I, p, p); D[1, 1] = 0.0
    β = (Xt' * Xt + 10^r.logλ * D + 1e-9I) \ (Xt' * y[train])
    rmse(y[valid], build(bank, t, valid) * β) + FEATURE_COST * nfeatures(r)
end

function mutate(rng, r::Recipe; rate = nothing)
    num, cat = copy(r.use_num), copy(r.use_cat)
    L = length(num) + length(cat)
    pr = something(rate, 2 / L)
    for i in eachindex(num); rand(rng) < pr && (num[i] = !num[i]); end
    for i in eachindex(cat); rand(rng) < pr && (cat[i] = !cat[i]); end
    logλ = clamp(r.logλ + (rand(rng) < 0.3 ? 0.3 * randn(rng) : 0.0), -2.0, 3.0)
    Recipe(num, cat, logλ)
end

function crossover(rng, a::Recipe, b::Recipe)
    num = BitVector(rand(rng, Bool) ? x : y for (x, y) in zip(a.use_num, b.use_num))
    cat = BitVector(rand(rng, Bool) ? x : y for (x, y) in zip(a.use_cat, b.use_cat))
    Recipe(num, cat, rand(rng, Bool) ? a.logλ : b.logλ)
end

"Start a population around a seed recipe."
function Population(seed::Recipe; size = 16, rng = MersenneTwister(7))
    Population(vcat([seed], [mutate(rng, seed; rate = 0.08) for _ in 2:size]), rng)
end

"""
Run `generations` rounds of tournament selection, crossover and mutation,
keeping the best two unchanged each round. Returns (best recipe, its fitness).
"""
function evolve!(pop::Population, bank, y, train, valid; generations = 3)
    score(r) = fitness(bank, y, r, train, valid)
    fits = score.(pop.members)
    for _ in 1:generations
        order = sortperm(fits)
        elite = pop.members[order[1:2]]
        pick() = (c = rand(pop.rng, eachindex(fits), 3); pop.members[c[argmin(fits[c])]])
        kids = [mutate(pop.rng, crossover(pop.rng, pick(), pick())) for _ in 1:length(pop.members)-2]
        pop.members = vcat(elite, kids)
        fits = vcat(fits[order[1:2]], score.(kids))
    end
    i = argmin(fits)
    pop.members[i], fits[i]
end
