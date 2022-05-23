################################################################################
# --------------- Base type for gradient-based generator:
################################################################################
"""
    AbstractGradientBasedGenerator

An abstract type that serves as the base type for gradient-based counterfactual generators. 
"""
abstract type AbstractGradientBasedGenerator <: AbstractGenerator end

# ----- Julia models -----
"""
    ∂ℓ(generator::AbstractGradientBasedGenerator, M::Union{Models.LogisticModel, Models.BayesianLogisticModel}, counterfactual_state::CounterfactualState)

The default method to compute the gradient of the loss function at the current counterfactual state for gradient-based generators. It assumes that `Zygote.jl` has gradient access.
"""
function ∂ℓ(generator::AbstractGradientBasedGenerator, M::Models.Models.AbstractDifferentiableJuliaModel, counterfactual_state::CounterfactualState)
    gradient(() -> ℓ(generator, counterfactual_state), params(counterfactual_state.x′))[counterfactual_state.x′]
end

# ----- RTorch model -----
using RCall
"""
    ∂ℓ(generator::AbstractGradientBasedGenerator, M::Models.RTorchModel, counterfactual_state::CounterfactualState)

The default method to compute the gradient of the loss function at the current counterfactual state for gradient-based generators. It assumes that `Zygote.jl` has gradient access.
"""
function ∂ℓ(generator::AbstractGradientBasedGenerator, M::Models.RTorchModel, counterfactual_state::CounterfactualState) 
    nn = M.nn
    x_cf = counterfactual_state.x′
    t = counterfactual_state.target_encoded
    R"""
    x <- torch_tensor($x_cf, requires_grad=TRUE)
    output <- $nn(x)
    obj_loss <- nnf_binary_cross_entropy_with_logits(output,$t)
    obj_loss$backward()
    """
    grad = rcopy(R"as_array(x$grad)")
    return grad
end

# ----- PyTorch model -----
using PyCall
"""
    ∂ℓ(generator::AbstractGradientBasedGenerator, M::Models.RTorchModel, counterfactual_state::CounterfactualState)

The default method to compute the gradient of the loss function at the current counterfactual state for gradient-based generators. It assumes that `Zygote.jl` has gradient access.
"""
function ∂ℓ(generator::AbstractGradientBasedGenerator, M::Models.PyTorchModel, counterfactual_state::CounterfactualState) 
    py"""
    import torch
    from torch import nn
    """
    nn = M.nn
    x′ = counterfactual_state.x′
    t = counterfactual_state.target_encoded
    x = reshape(x′, 1, length(x′))
    py"""
    x = torch.Tensor($x)
    x.requires_grad = True
    t = torch.Tensor($[t]).squeeze()
    output = $nn(x).squeeze()
    obj_loss = nn.BCEWithLogitsLoss()(output,t)
    obj_loss.backward()
    """
    grad = vec(py"x.grad.detach().numpy()")
    return grad
end

"""
    ∂h(generator::AbstractGradientBasedGenerator, counterfactual_state::CounterfactualState)

The default method to compute the gradient of the complexity penalty at the current counterfactual state for gradient-based generators. It assumes that `Zygote.jl` has gradient access.
"""
∂h(generator::AbstractGradientBasedGenerator, counterfactual_state::CounterfactualState) = gradient(() -> h(generator, counterfactual_state), params(counterfactual_state.x′))[counterfactual_state.x′]

# Gradient:
"""
    ∇(generator::AbstractGradientBasedGenerator, M::Models.AbstractDifferentiableModel, counterfactual_state::CounterfactualState)

The default method to compute the gradient of the counterfactual search objective for gradient-based generators. It simply computes the weighted sum over partial derivates. It assumes that `Zygote.jl` has gradient access.
"""
∇(generator::AbstractGradientBasedGenerator, M::Models.AbstractDifferentiableModel, counterfactual_state::CounterfactualState) = ∂ℓ(generator, M, counterfactual_state) + generator.λ * ∂h(generator, counterfactual_state)

"""
    generate_perturbations(generator::AbstractGradientBasedGenerator, counterfactual_state::CounterfactualState)

The default method to generate feature perturbations for gradient-based generators through simple gradient descent.
"""
function generate_perturbations(generator::AbstractGradientBasedGenerator, counterfactual_state::CounterfactualState) 
    𝐠ₜ = ∇(generator, counterfactual_state.M, counterfactual_state) # gradient
    Δx′ = - (generator.ϵ .* 𝐠ₜ) # gradient step
    return Δx′
end

"""
    mutability_constraints(generator::AbstractGradientBasedGenerator, counterfactual_state::CounterfactualState)

The default method to return mutability constraints that are dependent on the current counterfactual search state. For generic gradient-based generators, no state-dependent constraints are added.
"""
function mutability_constraints(generator::AbstractGradientBasedGenerator, counterfactual_state::CounterfactualState)
    mutability = counterfactual_state.params[:mutability]
    return mutability # no additional constraints for GenericGenerator
end 

"""
    conditions_satisified(generator::AbstractGradientBasedGenerator, counterfactual_state::CounterfactualState)

The default method to check if the all conditions for convergence of the counterfactual search have been satisified for gradient-based generators.
"""
function conditions_satisified(generator::AbstractGradientBasedGenerator, counterfactual_state::CounterfactualState)
    𝐠ₜ = ∇(generator, counterfactual_state.M, counterfactual_state)
    status = all(abs.(𝐠ₜ) .< generator.τ) 
    return status
end

##################################################
# Specific Generators
##################################################

include("GenericGenerator.jl") # Wachter et al. (2017)
include("GreedyGenerator.jl") # Schut et al. (2021)
include("REVISEGenerator.jl") # Joshi et al. (2019)