# generators.jl
#
# Core package functionality that implements algorithmic recourse.
module Generators

using ..CounterfactualState
using ..GenerativeModels
using Flux
using LinearAlgebra
using ..Losses
using ..Models

export AbstractGenerator, AbstractGradientBasedGenerator
export GenericGenerator, GenericGeneratorParams
export GreedyGenerator, GreedyGeneratorParams
export REVISEGenerator, REVISEGeneratorParams
export DiCEGenerator, DiCEGeneratorParams
export generator_catalog
export generate_perturbations, conditions_satisified, mutability_constraints   

"""
    AbstractGenerator

An abstract type that serves as the base type for counterfactual generators. 
"""
abstract type AbstractGenerator end

# Loss:
"""
    ℓ(generator::AbstractGenerator, counterfactual_state::CounterfactualState.State)

The default method to apply the generator loss function to the current counterfactual state for any generator.
"""
function ℓ(generator::AbstractGenerator, counterfactual_state::CounterfactualState.State)

    loss_fun = !isnothing(generator.loss) ? getfield(Losses, generator.loss) : CounterfactualState.guess_loss(counterfactual_state)
    @assert !isnothing(loss_fun) "No loss function provided and loss function could not be guessed based on model."
    loss = loss_fun(
        getfield(Models, :logits)(counterfactual_state.M, counterfactual_state.f(counterfactual_state.s′)), 
        counterfactual_state.target_encoded
    )    
    return loss
end

# Complexity:
"""
    h(generator::AbstractGenerator, counterfactual_state::CounterfactualState.State)

The default method to apply the generator complexity penalty to the current counterfactual state for any generator.
"""
function h(generator::AbstractGenerator, counterfactual_state::CounterfactualState.State)
    dist_ = generator.complexity(
        counterfactual_state.x .- counterfactual_state.f(counterfactual_state.s′)
    )
    penalty = generator.λ * dist_
    return penalty
end

include("gradient_based/functions.jl")

generator_catalog = Dict(
    :generic => Generators.GenericGenerator,
    :greedy => Generators.GreedyGenerator,
    :revise => Generators.REVISEGenerator,
    :dice => Generators.DiCEGenerator
)

end