"""
	Flux.Losses.logitbinarycrossentropy(ce::AbstractCounterfactualExplanation)

Simply extends the `logitbinarycrossentropy` method to work with objects of type `AbstractCounterfactualExplanation`.
"""
function Flux.Losses.logitbinarycrossentropy(
    ce::AbstractCounterfactualExplanation; kwargs...
)
    loss = Flux.Losses.logitbinarycrossentropy(
        logits(ce.M, CounterfactualExplanations.decode_state(ce)),
        ce.target_encoded;
        kwargs...,
    )
    return loss
end

"""
	Flux.Losses.logitcrossentropy(ce::AbstractCounterfactualExplanation)

Simply extends the `logitcrossentropy` method to work with objects of type `AbstractCounterfactualExplanation`.
"""
function Flux.Losses.logitcrossentropy(ce::AbstractCounterfactualExplanation; kwargs...)
    loss = Flux.Losses.logitcrossentropy(
        logits(ce.M, CounterfactualExplanations.decode_state(ce)),
        ce.target_encoded;
        kwargs...,
    )
    return loss
end

"""
	Flux.Losses.mse(ce::AbstractCounterfactualExplanation)

Simply extends the `mse` method to work with objects of type `AbstractCounterfactualExplanation`.
"""
function Flux.Losses.mse(ce::AbstractCounterfactualExplanation; kwargs...)
    loss = Flux.Losses.mse(
        logits(ce.M, CounterfactualExplanations.decode_state(ce)),
        ce.target_encoded;
        kwargs...,
    )
    return loss
end

"""
    predictive_entropy(ce::AbstractCounterfactualExplanation; agg=Statistics.mean)

Computes the predictive entropy of the counterfactuals.
Explained in https://arxiv.org/abs/1406.2541.
"""
function predictive_entropy(ce::AbstractCounterfactualExplanation; agg=Statistics.mean)
    model = ce.M
    counterfactual_data = ce.data
    X = CounterfactualExplanations.decode_state(ce)
    p = CounterfactualExplanations.Models.predict_proba(model, counterfactual_data, X)
    output = -agg(sum(@.(p * log(p)); dims=2))
    return output
end

function conformal_training_loss(
    ce::AbstractCounterfactualExplanation;
    temp::Real = 0.1,
    agg = mean,
    kwargs...,
)
    conf_model = ce.M.model
    fitresult = ce.M.likelihood
    X = CounterfactualExplanations.decode_state(ce)
    y = ce.target_encoded[:, :, 1]
    if ce.M.likelihood == :classification_binary
        y = binary_to_onehot(y)
    end
    y = permutedims(y)

    n_classes = length(ce.data.y_levels)
    loss_mat = ones(n_classes, n_classes)
    loss = map(eachslice(X, dims = ndims(X))) do x
        x = ndims(x) == 1 ? x[:, :]' : x
        ConformalPrediction.ConformalTraining.classification_loss(
            conf_model,
            fitresult,
            x,
            y;
            temp = temp,
            loss_matrix = loss_mat,
        )
    end
    loss = agg(loss)[1]
    return loss
end
