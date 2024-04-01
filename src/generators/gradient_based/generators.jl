const default_distance = Objectives.distance_l1

"Constructor for `GenericGenerator`."
function GenericGenerator(; λ::AbstractFloat=0.1, kwargs...)
    return GradientBasedGenerator(; penalty=default_distance, λ=λ, kwargs...)
end

"Constructor for `WachterGenerator`."
function WachterGenerator(; λ::AbstractFloat=0.1, kwargs...)
    return GradientBasedGenerator(; penalty=Objectives.distance_mad, λ=λ, kwargs...)
end

"Constructor for `DiCEGenerator`."
function DiCEGenerator(; λ::Vector{<:AbstractFloat}=[0.1, 0.1], kwargs...)
    _penalties = [default_distance, Objectives.ddp_diversity]
    return GradientBasedGenerator(; penalty=_penalties, λ=λ, kwargs...)
end

"Constructor for `ClaPGenerator`."
function ClaPROARGenerator(; λ::Vector{<:AbstractFloat}=[0.1, 0.5], kwargs...)
    _penalties = [default_distance, Objectives.model_loss_penalty]
    return GradientBasedGenerator(; penalty=_penalties, λ=λ, kwargs...)
end

"Constructor for `GravitationalGenerator`."
function GravitationalGenerator(; λ::Vector{<:AbstractFloat}=[0.1, 0.5], kwargs...)
    _penalties = [default_distance, Objectives.distance_from_target]
    return GradientBasedGenerator(; penalty=_penalties, λ=λ, kwargs...)
end

"Constructor for `REVISEGenerator`."
function REVISEGenerator(; λ::AbstractFloat=0.1, latent_space=true, kwargs...)
    return GradientBasedGenerator(;
        penalty=default_distance, λ=λ, latent_space=latent_space, kwargs...
    )
end

"Constructor for `GreedyGenerator`."
function GreedyGenerator(; η=0.1, n=nothing, kwargs...)
    opt = CounterfactualExplanations.Generators.JSMADescent(; η=η, n=n)
    return GradientBasedGenerator(; penalty=default_distance, λ=0.0, opt=opt, kwargs...)
end

"Constructor for `CLUEGenerator`."
function CLUEGenerator(; λ::AbstractFloat=0.1, latent_space=true, kwargs...)
    return GradientBasedGenerator(;
        loss=predictive_entropy,
        penalty=default_distance,
        λ=λ,
        latent_space=latent_space,
        kwargs...,
    )
end

"Constructor for `ECECCCoGenerator`: Energy Constrained Conformal Counterfactual Explanation Generator."
function ECCCoGenerator(;
    λ::Union{AbstractFloat,Vector{<:AbstractFloat}} = [0.2, 0.4, 0.4],
    κ::Real = 1.0,
    temp::Real = 0.1,
    opt::Union{Nothing,Flux.Optimise.AbstractOptimiser} = nothing,
    use_class_loss::Bool = false,
    use_energy_delta::Bool = false,
    nsamples::Union{Nothing,Int} = nothing,
    nmin::Union{Nothing,Int} = nothing,
    niter::Union{Nothing,Int} = nothing,
    reg_strength::Real = 0.1,
    decay::Tuple = (0.1, 1),
    dim_reduction::Bool = false,
    kwargs...,
)

    # Default ECCCo parameters
    nsamples = isnothing(nsamples) ? 10 : nsamples
    nmin = isnothing(nmin) ? 1 : nmin
    niter = isnothing(niter) ? 100 : niter

    # Default optimiser
    if isnothing(opt)
        opt = CounterfactualExplanations.Generators.Descent(0.1)
    end

    # Loss function
    if use_class_loss
        loss_fun(ce::AbstractCounterfactualExplanation) =
            conformal_training_loss(ce; temp = temp)
    else
        loss_fun = nothing
    end

    _energy_penalty =
        use_energy_delta ?
        (
            Objectives.energy_delta,
            (
                n = nsamples,
                nmin = nmin,
                niter = niter,
                reg_strength = reg_strength,
                decay = decay,
            ),
        ) : (Objectives.distance_from_energy, (n = nsamples, nmin = nmin, niter = niter))

    _penalties = [
        (Objectives.distance_l1, []),
        (Objectives.set_size_penalty, (κ = κ, temp = temp)),
        _energy_penalty,
    ]
    λ = λ isa AbstractFloat ? [0.0, λ, λ] : λ

    # Generator
    return GradientBasedGenerator(;
        loss = loss_fun,
        penalty = _penalties,
        λ = λ,
        opt = opt,
        dim_reduction = dim_reduction,
        kwargs...,
    )
end

"Constructor for `ProbeGenerator`."
function ProbeGenerator(;
    λ::AbstractFloat=0.1,
    loss::Symbol=:logitbinarycrossentropy,
    penalty=Objectives.distance_l1,
    kwargs...,
)
    @assert haskey(losses_catalogue, loss) "Loss function not found in catalogue."
    user_loss = Objectives.losses_catalogue[loss]
    return GradientBasedGenerator(; loss=user_loss, penalty=penalty, λ=λ, kwargs...)
end
