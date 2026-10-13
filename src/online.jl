"""
Online ridge regression by recursive least squares (RLS).

Batch ridge solves  β = (XᵀX + λD)⁻¹ Xᵀy  with D = I except D[1,1] = 0 (the
intercept is not penalized). Keep P = (XᵀX + λD)⁻¹ and β. When one more house
(x, y) arrives, the Sherman–Morrison identity updates both exactly:

    k = P x / (1 + xᵀ P x)
    β ← β + k (y − xᵀβ)
    P ← P − k xᵀ P

After absorbing every row this is the same β as refitting batch ridge on all
of them (checked in the tests), but each sale costs O(p²) instead of a refit.
"""

using LinearAlgebra

mutable struct OnlineRidge
    β::Vector{Float64}
    P::Matrix{Float64}
    n::Int
end

function OnlineRidge(X::AbstractMatrix, y::AbstractVector, λ::Real)
    p = size(X, 2)
    D = Matrix{Float64}(I, p, p); D[1, 1] = 0.0
    P = inv(Symmetric(X' * X + λ * D + 1e-9I))
    OnlineRidge(P * (X' * y), Matrix(P), size(X, 1))
end

function update!(m::OnlineRidge, x::AbstractVector, y::Real)
    Px = m.P * x
    k = Px ./ (1 + dot(x, Px))
    m.β .+= k .* (y - dot(x, m.β))
    m.P .-= k * Px'
    m.n += 1
    m
end

function update!(m::OnlineRidge, X::AbstractMatrix, y::AbstractVector)
    for i in axes(X, 1); update!(m, view(X, i, :), y[i]); end
    m
end

predict(m::OnlineRidge, X::AbstractMatrix) = X * m.β
