"""
    distance_mad(ce::AbstractCounterfactualExplanation; agg=mean)

This is the distance measure proposed by Wachter et al. (2017).
"""
function distance_mad(
    ce::AbstractCounterfactualExplanation; agg=Statistics.mean, noise=1e-5, kwrgs...
)
    X = ce.data.X
    mad = []
    ChainRulesCore.ignore_derivatives() do
        _dict = ce.search
        if !(:mad_features ∈ collect(keys(_dict)))
            X̄ = Statistics.median(X; dims=ndims(X))
            _mad = Statistics.median(abs.(X .- X̄); dims=ndims(X))
            _dict[:mad_features] = _mad .+ size(X, 1) * noise        # add noise to avoid division by zero
        end
        _mad = _dict[:mad_features]
        push!(mad, _mad)
    end
    return distance(ce; agg=agg, weights=1.0 ./ mad[1], kwrgs...)
end

"""
    distance_l0(ce::AbstractCounterfactualExplanation)

Computes the L0 distance of the counterfactual to the original factual.
"""
function distance_l0(ce::AbstractCounterfactualExplanation; kwrgs...)
    return distance(ce; p=0, kwrgs...)
end

"""
    distance_l1(ce::AbstractCounterfactualExplanation)

Computes the L1 distance of the counterfactual to the original factual.
"""
function distance_l1(ce::AbstractCounterfactualExplanation; kwrgs...)
    return distance(ce; p=1, kwrgs...)
end

"""
    distance_l2(ce::AbstractCounterfactualExplanation)

Computes the L2 (Euclidean) distance of the counterfactual to the original factual.
"""
function distance_l2(ce::AbstractCounterfactualExplanation; kwrgs...)
    return distance(ce; p=2, kwrgs...)
end

"""
    distance_linf(ce::AbstractCounterfactualExplanation)

Computes the L-inf distance of the counterfactual to the original factual.
"""
function distance_linf(ce::AbstractCounterfactualExplanation; kwrgs...)
    return distance(ce; p=Inf, kwrgs...)
end

"""
    ddp_diversity(
        ce::AbstractCounterfactualExplanation;
        perturbation_size=1e-5
    )

Evaluates how diverse the counterfactuals are using a Determinantal Point Process (DDP).
"""
function ddp_diversity(
    ce::AbstractCounterfactualExplanation; perturbation_size=1e-3, agg=det
)
    X = ce.s′
    xs = eachslice(X; dims=ndims(X))
    K = [1 / (1 + LinearAlgebra.norm(x .- y)) for x in xs, y in xs]
    K += LinearAlgebra.Diagonal(
        Random.randn(eltype(X), size(X)[end]) * convert(eltype(X), perturbation_size)
    )
    cost = -agg(K)
    return cost
end

function distance_from_targets(
    ce::AbstractCounterfactualExplanation;
    n::Int=1000,
    agg=mean,
    n_nearest_neighbors::Union{Int,Nothing}=nothing,
)
    target_idx = ce.data.output_encoder.labels .== ce.target
    target_samples = ce.data.X[:, target_idx] |> X -> X[:, rand(1:end, n)]
    x′ = CounterfactualExplanations.counterfactual(ce)
    loss = map(eachslice(x′; dims=ndims(x′))) do x
        Δ = map(eachcol(target_samples)) do xsample
            norm(x - xsample, 1)
        end
        if !isnothing(n_nearest_neighbors)
            Δ = sort(Δ)[1:n_nearest_neighbors]
        end
        return mean(Δ)
    end
    loss = agg(loss)[1]

    return loss
end

"""
    set_size_penalty(ce::AbstractCounterfactualExplanation)

Penalty for smooth conformal set size.
"""
function set_size_penalty(
    ce::AbstractCounterfactualExplanation;
    κ::Real = 1.0,
    temp::Real = 0.1,
    agg = mean,
)

    _loss = 0.0

    conf_model = ce.M.model
    fitresult = ce.M.fitresult
    X = CounterfactualExplanations.decode_state(ce)
    _loss = map(eachslice(X, dims = ndims(X))) do x
        x = ndims(x) == 1 ? x[:, :] : x
        if target_probs(ce, x)[1] >= 0.5
            l = ConformalPrediction.ConformalTraining.smooth_size_loss(
                conf_model,
                fitresult,
                x';
                κ = κ,
                temp = temp,
            )[1]
        else
            l = 0.0
        end
        return l
    end
    _loss = agg(_loss)

    return _loss

end

function energy_delta(
    ce::AbstractCounterfactualExplanation;
    n::Int = 50,
    niter = 500,
    from_buffer = true,
    agg = mean,
    choose_lowest_energy = true,
    choose_random = false,
    nmin::Int = 25,
    return_conditionals = false,
    reg_strength = 0.1,
    decay::Tuple = (0.1, 1),
    kwargs...,
)

    xproposed = CounterfactualExplanations.decode_state(ce)     # current state
    t = get_target_index(ce.data.y_levels, ce.target)
    E(x) = -logits(ce.M, x)[t, :]                                # negative logits for taraget class

    # Generative loss:
    gen_loss = E(xproposed)
    gen_loss = reduce((x, y) -> x + y, gen_loss) / length(gen_loss)                  # aggregate over samples

    # Regularization loss:
    reg_loss = norm(E(xproposed))^2
    reg_loss = reduce((x, y) -> x + y, reg_loss) / length(reg_loss)                  # aggregate over samples

    # Decay:
    iter = total_steps(ce)
    ϕ = 1.0
    if iter % decay[2] == 0
        ϕ = exp(-decay[1] * total_steps(ce))
    end

    # Total loss:
    ℒ = ϕ * (gen_loss + reg_strength * reg_loss)

    return ℒ

end

"""
    distance_from_energy(ce::AbstractCounterfactualExplanation)

Computes the distance from the counterfactual to generated conditional samples.
"""
function distance_from_energy(
    ce::AbstractCounterfactualExplanation;
    n::Int = 50,
    niter = 500,
    from_buffer = true,
    agg = mean,
    choose_lowest_energy = true,
    choose_random = false,
    nmin::Int = 25,
    return_conditionals = false,
    p::Int = 1,
    kwargs...,
)

    _loss = 0.0
    nmin = minimum([nmin, n])

    @assert choose_lowest_energy ⊻ choose_random || !choose_lowest_energy && !choose_random "Must choose either lowest energy or random samples or neither."

    conditional_samples = []
    ignore_derivatives() do
        _dict = ce.params
        if !(:energy_sampler ∈ collect(keys(_dict)))
            _dict[:energy_sampler] =
                ECCCo.EnergySampler(ce; niter = niter, nsamples = n, kwargs...)
        end
        eng_sampler = _dict[:energy_sampler]
        if choose_lowest_energy
            nmin = minimum([nmin, size(eng_sampler.buffer)[end]])
            xmin = ECCCo.get_lowest_energy_sample(eng_sampler; n = nmin)
            push!(conditional_samples, xmin)
        elseif choose_random
            push!(conditional_samples, rand(eng_sampler, n; from_buffer = from_buffer))
        else
            push!(conditional_samples, eng_sampler.buffer)
        end
    end

    _loss = map(eachcol(conditional_samples[1])) do xsample
        distance(ce; from = xsample, agg = agg, p = p)
    end
    _loss = reduce((x, y) -> x + y, _loss) / n       # aggregate over samples

    if return_conditionals
        return conditional_samples[1]
    end
    return _loss

end


