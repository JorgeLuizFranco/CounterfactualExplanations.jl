# recourse.jl

struct Recourse
    x̲::Vector{Float64}
    y̲::Float64
    path::Matrix{Float64}
    generator::Generator
    immutable::AbstractArray
end;

# --------------- Outer constructor methods: 
# Check if recourse is valid:
function valid(recourse::Recourse; 𝓜=nothing)
    if isnothing(𝓜)
        valid = recourse.y̲ == recourse.target
    else 
        valid = 𝓜(recourse.x̲) == recourse.target
    end
    return valid
end

# Compute cost associated with counterfactual:
function cost(recourse::Recourse; cost_fun=nothing, cost_fun_kargs)
    return cost_fun(recourse.generator.x̅, recourse.x̲; cost_fun_kargs...)
end