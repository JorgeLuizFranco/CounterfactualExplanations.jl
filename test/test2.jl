#__precompile__(false)


using CategoricalArrays
using CounterfactualExplanations
using CounterfactualExplanations.DataPreprocessing

using TaijaData
using Plots
#using TaijaPlotting
using Flux

using MLUtils: stack

using DataAPI
using Distributions: pdf
using NearestNeighborModels: KNNClassifier

function predict_proba(
    M::AbstractFittedModel,
    counterfactual_data::Union{Nothing,CounterfactualData},
    X::Union{Nothing,AbstractArray},
)
    @assert !(isnothing(counterfactual_data) && isnothing(X))
    X = isnothing(X) ? counterfactual_data.X : X
    p = probs(M, X)
    binary = M.likelihood == :classification_binary
    p = binary ? binary_to_onehot(p) : p
    return p
end

function Plots.plot(
    M::AbstractFittedModel,
    data::DataPreprocessing.CounterfactualData;
    target::Union{Nothing,RawTargetType} = nothing,
    colorbar = true,
    title = "",
    length_out = 100,
    zoom = -0.1,
    xlims = nothing,
    ylims = nothing,
    linewidth = 0.1,
    alpha = 1.0,
    contour_alpha = 1.0,
    dim_red::Symbol = :pca,
    kwargs...,
)
    X, _ = DataPreprocessing.unpack_data(data)
    ŷ = probs(M, X) # true predictions
    if size(ŷ, 1) > 1
        ŷ = vec(Flux.onecold(ŷ, 1:size(ŷ, 1)))
    else
        ŷ = vec(ŷ)
    end

    X, y, multi_dim = prepare_for_plotting(data; dim_red = dim_red)

    # Surface range:
    zoom = zoom * maximum(abs.(X))
    if isnothing(xlims)
        xlims = (minimum(X[:, 1]), maximum(X[:, 1])) .+ (zoom, -zoom)
    else
        xlims = xlims .+ (zoom, -zoom)
    end
    if isnothing(ylims)
        ylims = (minimum(X[:, 2]), maximum(X[:, 2])) .+ (zoom, -zoom)
    else
        ylims = ylims .+ (zoom, -zoom)
    end
    x_range = convert.(eltype(X), range(xlims[1]; stop = xlims[2], length = length_out))
    y_range = convert.(eltype(X), range(ylims[1]; stop = ylims[2], length = length_out))

    if multi_dim
        knn1, y_train = voronoi(X, ŷ)
        predict_ =
            (X::AbstractVector) -> vec(
                pdf(
                    MLJBase.predict(knn1, MLJBase.table(reshape(X, 1, 2))),
                    DataAPI.levels(y_train),
                ),
            )
        Z = [predict_([x, y]) for x in x_range, y in y_range]
    else
        predict_ = function (X::AbstractVector)
            X = permutedims(permutedims(X))
            z = predict_proba(M, data, X)
            return z
        end
        Z = [predict_([x, y]) for x in x_range, y in y_range]
    end

    # Pre-processes:
    Z = reduce(hcat, Z)
    if isnothing(target)
        target = data.y_levels[1]
        if size(Z, 1) > 2
            @info "No target label supplied, using first."
        end
    end
    target_idx = get_target_index(data.y_levels, target)

    # Contour:
    Plots.contourf(
        x_range,
        y_range,
        Z[Int(target_idx), :];
        colorbar = colorbar,
        title = title,
        linewidth = linewidth,
        xlims = xlims,
        ylims = ylims,
        kwargs...,
        alpha = contour_alpha,
    )

    # Samples:
    return Plots.scatter!(data; dim_red = dim_red, alpha = alpha, kwargs...)
end

function voronoi(X::AbstractMatrix, y::AbstractVector)
    knnc = KNNClassifier(; K = 1) # KNNClassifier instantiation
    X = MLJBase.table(X)
    y = CategoricalArrays.categorical(y)
    knnc_mach = MLJBase.machine(knnc, X, y)
    MLJBase.fit!(knnc_mach)
    return knnc_mach, y
end

"""
    Plots.plot(
        ce::CounterfactualExplanation;
        alpha_ = 0.5,
        plot_up_to::Union{Nothing,Int} = nothing,
        plot_proba::Bool = false,
        kwargs...,
    )

Calling `plot` on an instance of type `CounterfactualExplanation` returns a plot that visualises the entire counterfactual path. For multi-dimensional input data, the data is first compressed into two dimensions. The decision boundary is then approximated using using a Nearest Neighbour classifier. This is still somewhat experimental at the moment.


# Examples

```julia-repl
# Search:
generator = GenericGenerator()
ce = generate_counterfactual(x, target, counterfactual_data, M, generator)

plot(ce)
```
"""
function Plots.plot(
    ce_plot::CounterfactualExplanation;
    alpha_ = 0.5,
    plot_up_to::Union{Nothing,Int} = nothing,
    plot_proba::Bool = false,
    n_points = 1000,
    kwargs...,
)

    ce = deepcopy(ce_plot)
    ce.data = DataPreprocessing.subsample(ce.data, n_points)

    max_iter = total_steps(ce)
    max_iter = if isnothing(plot_up_to)
        total_steps(ce)
    else
        minimum([plot_up_to, max_iter])
    end
    max_iter += 1
    ingredients = set_up_plots(ce; alpha = alpha_, plot_proba = plot_proba, kwargs...)

    for t = 1:max_iter
        final_state = t == max_iter
        plot_state(ce, t, final_state; ingredients...)
    end

    plt = if plot_proba
        Plots.plot(ingredients.p1, ingredients.p2; kwargs...)
    else
        Plots.plot(ingredients.p1; kwargs...)
    end

    return plt
end

"""
    animate_path(ce::CounterfactualExplanation, path=tempdir(); plot_proba::Bool=false, kwargs...)

Returns and animation of the counterfactual path.

# Examples

```julia-repl
# Search:
generator = GenericGenerator()
ce = generate_counterfactual(x, target, counterfactual_data, M, generator)

animate_path(ce)
```
"""
function animate_path(
    ce::CounterfactualExplanation,
    path = tempdir();
    alpha_ = 0.5,
    plot_up_to::Union{Nothing,Int} = nothing,
    plot_proba::Bool = false,
    kwargs...,
)
    max_iter = total_steps(ce)
    max_iter = if isnothing(plot_up_to)
        total_steps(ce)
    else
        minimum([plot_up_to, max_iter])
    end
    max_iter += 1
    ingredients = set_up_plots(ce; alpha = alpha_, plot_proba = plot_proba, kwargs...)

    anim = @animate for t = 1:max_iter
        final_state = t == max_iter
        plot_state(ce, t, final_state; ingredients...)
        if plot_proba
            plot(ingredients.p1, ingredients.p2; kwargs...)
        else
            plot(ingredients.p1; kwargs...)
        end
    end
    return anim
end

"""
    plot_state(
        ce::CounterfactualExplanation,
        t::Int,
        final_state::Bool;
        kwargs...
    )

Helper function that plots a single step of the counterfactual path.
"""
function plot_state(ce::CounterfactualExplanation, t::Int, final_state::Bool; kwargs...)
    args = PlotIngredients(; kwargs...)
    x1 = args.path_embedded[1, t, :]
    x2 = args.path_embedded[2, t, :]
    y = args.path_labels[t]
    _c = CategoricalArrays.levelcode.(y)
    n_ = ce.num_counterfactuals
    label_ = reshape(["C$i" for i = 1:n_], 1, n_)
    if !final_state
        scatter!(args.p1, x1, x2; group = y, colour = _c, ms = 5, label = "")
    else
        scatter!(args.p1, x1, x2; group = y, colour = _c, ms = 10, label = "")
        if n_ > 1
            label_1 = vec([text(lab, 5) for lab in label_])
            annotate!(x1, x2, label_1)
        end
    end
    if args.plot_proba
        probs_ = reshape(reduce(vcat, args.path_probs[1:t]), t, n_)
        if t == 1 && n_ > 1
            label_2 = label_
        else
            label_2 = ""
        end
        plot!(
            args.p2,
            probs_;
            label = label_2,
            color = reshape(1:n_, 1, n_),
            title = "p(y=$(ce.target))",
        )
    end
end

"A container used for plotting."
Base.@kwdef struct PlotIngredients
    p1::Any
    p2::Any
    path_embedded::Any
    path_labels::Any
    path_probs::Any
    alpha::Any
    plot_proba::Any
end

"""
    set_up_plots(
        ce::CounterfactualExplanation;
        alpha,
        plot_proba,
        kwargs...
    )

A helper method that prepares data for plotting.
"""
function set_up_plots(ce::CounterfactualExplanation; alpha, plot_proba, kwargs...)
    p1 = plot(ce.M, ce.data; target = ce.target, alpha = alpha, kwargs...)
    p2 = plot(; xlims = (1, total_steps(ce) + 1), ylims = (0, 1))
    path_embedded = embed_path(ce)
    path_labels = CounterfactualExplanations.counterfactual_label_path(ce)
    y_levels = ce.data.y_levels
    path_labels = map(x -> CategoricalArrays.categorical(x; levels = y_levels), path_labels)
    path_probs = CounterfactualExplanations.target_probs_path(ce)
    output = (
        p1 = p1,
        p2 = p2,
        path_embedded = path_embedded,
        path_labels = path_labels,
        path_probs = path_probs,
        alpha = alpha,
        plot_proba = plot_proba,
    )
    return output
end

function embed(data::CounterfactualData, X::AbstractArray = nothing; dim_red::Symbol = :pca)

    # Training compressor:
    if isnothing(data.compressor)
        X_train, _ = DataPreprocessing.unpack_data(data)
        if size(X_train, 1) < 3
            tfn = data.compressor
        else
            @info "Training model to compress data."
            if dim_red == :pca
                tfn = MultivariateStats.fit(PCA, X_train; maxoutdim = 2)
            elseif dim_red == :tsne
                tfn = MultivariateStats.fit(TSNE, X_train; maxoutdim = 2)
            end
            data.compressor = nothing
            X = isnothing(X) ? X_train : X
        end
    else
        tfn = data.compressor
    end

    # Transforming:
    X = typeof(X) <: Vector{<:Matrix} ? hcat(X...) : X
    if !isnothing(tfn) && !isnothing(X)
        X = MultivariateStats.predict(tfn, X)
    else
        X = isnothing(X) ? X_train : X
    end

    return X
end

"""
    embed_path(ce::CounterfactualExplanation)

Helper function that embeds path into two dimensions for plotting.
"""
function embed_path(ce::CounterfactualExplanation)
    data_ = ce.data
    return embed(data_, path(ce))
end

function prepare_for_plotting(data::CounterfactualData; dim_red::Symbol = :pca)
    X, _ = DataPreprocessing.unpack_data(data)
    y = data.output_encoder.labels
    @assert size(X, 1) != 1 "Don't know how to plot 1-dimensional data."
    multi_dim = size(X, 1) > 2
    if multi_dim
        X = embed(data, X; dim_red = dim_red)
    end
    return X', y, multi_dim
end

function Plots.scatter!(data::CounterfactualData; dim_red::Symbol = :pca, kwargs...)
    X, y, _ = prepare_for_plotting(data; dim_red = dim_red)
    _c = Int.(y.refs)
    return Plots.scatter!(X[:, 1], X[:, 2]; group = y, colour = _c, kwargs...)
end

function set_up_plots(ce::CounterfactualExplanation; alpha, plot_proba, kwargs...)
    p1 = plot(ce.M, ce.data; target = ce.target, alpha = alpha, kwargs...)
    p2 = plot(; xlims = (1, total_steps(ce) + 1), ylims = (0, 1))
    path_embedded = embed_path(ce)
    path_labels = CounterfactualExplanations.counterfactual_label_path(ce)
    y_levels = ce.data.y_levels
    path_labels = map(x -> CategoricalArrays.categorical(x; levels = y_levels), path_labels)
    path_probs = CounterfactualExplanations.target_probs_path(ce)
    output = (
        p1 = p1,
        p2 = p2,
        path_embedded = path_embedded,
        path_labels = path_labels,
        path_probs = path_probs,
        alpha = alpha,
        plot_proba = plot_proba,
    )
    return output
end

function Plots.plot(
    ce_plot::CounterfactualExplanation;
    alpha_ = 0.5,
    plot_up_to::Union{Nothing,Int} = nothing,
    plot_proba::Bool = false,
    n_points = 1000,
    kwargs...,
)

    ce = deepcopy(ce_plot)
    ce.data = DataPreprocessing.subsample(ce.data, n_points)

    max_iter = total_steps(ce)
    max_iter = if isnothing(plot_up_to)
        total_steps(ce)
    else
        minimum([plot_up_to, max_iter])
    end
    max_iter += 1
    ingredients = set_up_plots(ce; alpha = alpha_, plot_proba = plot_proba, kwargs...)

    for t = 1:max_iter
        final_state = t == max_iter
        plot_state(ce, t, final_state; ingredients...)
    end

    plt = if plot_proba
        Plots.plot(ingredients.p1, ingredients.p2; kwargs...)
    else
        Plots.plot(ingredients.p1; kwargs...)
    end

    return plt
end

# Counteractual data and model:
counterfactual_data = CounterfactualData(TaijaData.load_linearly_separable()...)
M = fit_model(counterfactual_data, :Linear)
target = 2
factual = 1
chosen = rand(findall(predict_label(M, counterfactual_data) .== factual))
x = select_factual(counterfactual_data, chosen)

# Search:
generator = WachterGenerator()
ce = generate_counterfactual(x, target, counterfactual_data, M, generator)

print(ce)

Plots.plot(ce)

savefig("test.png")

# include("encodings.jl")
# include("generate_counterfactual.jl")
# include("path_tracking.jl")
# include("utils.jl")
