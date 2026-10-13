"""
Linear regression written from scratch, three ways. All three fit the same
model, ŷ = Xβ, and should land on the same β (the first two exactly, gradient
descent to a few decimals).

1. Ordinary least squares via the normal equation
       β = (XᵀX)⁻¹ Xᵀy
   `fit_ols_normal` writes that formula out literally. `fit_ols` solves the
   same system with Julia's `\\` (a QR factorization), which is what you use
   in practice: same answer, better numerics, no explicit inverse.

2. Ridge regression
       β = (XᵀX + λI)⁻¹ Xᵀy
   One extra term. λ shrinks the coefficients toward zero, which trades a
   little bias for less variance (less overfitting). The intercept is not
   penalized, so I[1,1] = 0.

3. Gradient descent
   Start at β = 0 and repeatedly step against the gradient of the loss:
       loss(β)  = (1/2n) ‖Xβ − y‖²
       ∇loss(β) = (1/n) Xᵀ(Xβ − y)
       β       ← β − lr · ∇loss(β)
   Slower and approximate, but it is the method that scales to models with
   no closed form, which is every neural network.
"""

using LinearAlgebra

struct LinearModel
    β::Vector{Float64}
end

predict(m::LinearModel, X::AbstractMatrix) = X * m.β

"Normal equation, written out. Fails loudly if XᵀX is singular."
fit_ols_normal(X, y) = LinearModel(inv(X' * X) * (X' * y))

"Normal equation solved by QR. The production version of the line above."
fit_ols(X, y) = LinearModel(X \ y)

"Ridge. λ = 0 reduces to OLS."
function fit_ridge(X, y, λ::Real)
    p = size(X, 2)
    I_nointercept = Matrix{Float64}(I, p, p)
    I_nointercept[1, 1] = 0.0
    LinearModel((X' * X + λ * I_nointercept) \ (X' * y))
end

"Gradient descent. Returns the model and the loss after every epoch."
function fit_gd(X, y; lr = 0.1, epochs = 500)
    n, p = size(X)
    β = zeros(p)
    history = Float64[]
    for _ in 1:epochs
        r = X * β .- y                       # residuals
        push!(history, sum(abs2, r) / (2n))  # loss
        β .-= lr .* (X' * r) ./ n            # one step downhill
    end
    LinearModel(β), history
end
