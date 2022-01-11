# --------------- Base type for model:
module Models

using Flux

abstract type FittedModel end


# -------- Linear model:
# This is an example of how to construct a FittedModel subtype:
struct LogisticModel <: FittedModel
    w::AbstractArray
    b::AbstractArray
end

# What follows are the two required outer methods:
logits(𝓜::LogisticModel, X::AbstractArray) = X * 𝓜.w .+ 𝓜.b
probs(𝓜::LogisticModel, X::AbstractArray) = Flux.σ.(logits(𝓜, X))

# -------- Bayesian model:
struct BayesianLogisticModel <: FittedModel
end
end