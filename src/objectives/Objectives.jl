module Objectives

using ..CounterfactualExplanations
using Flux
using Flux.Losses
using ChainRulesCore
using LinearAlgebra
using Statistics
using Random

include("distance_utils.jl")
include("loss_functions.jl")
include("penalties.jl")

export logitbinarycrossentropy, logitcrossentropy, mse, predictive_entropy, conformal_training_loss
export losses_catalogue
export distance, distance_mad, distance_l0, distance_l1, distance_l2, distance_linf
export ddp_diversity
export set_size_penalty, distance_from_energy, energy_delta
export penalties_catalogue

const losses_catalogue = Dict(
    :logitbinarycrossentropy => logitbinarycrossentropy,
    :logitcrossentropy => logitcrossentropy,
    :mse => mse,
)

const penalties_catalogue = Dict(
    :distance_mad => distance_mad,
    :distance_l0 => distance_l0,
    :distance_l1 => distance_l1,
    :distance_l2 => distance_l2,
    :distance_linf => distance_linf,
    :ddp_diversity => ddp_diversity,
    :set_size_penalty => set_size_penalty,
    :distance_from_energy => distance_from_energy,
    :energy_delta => energy_delta,
)

end
