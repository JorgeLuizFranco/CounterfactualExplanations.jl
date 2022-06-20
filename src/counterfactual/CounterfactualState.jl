module CounterfactualState

using ..Models

################################################################################
# --------------- Constructor for counterfactual state:
################################################################################
struct State
    x::AbstractArray
    s′::AbstractArray
    f::Function
    target_encoded::Union{Number, AbstractVector}
    M::AbstractFittedModel
    params::Dict
    search::Union{Dict,Nothing}
end

end
