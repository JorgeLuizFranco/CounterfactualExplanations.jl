module CounterfactualExplanations

# Dependencies:
using Flux
using LinearAlgebra

include("data/Data.jl")
using .Data

include("data_preprocessing/DataPreprocessing.jl")
using .DataPreprocessing
export CounterfactualData, select_factual

include("models/Models.jl")
using .Models
export AbstractFittedModel, LogisticModel, BayesianLogisticModel, probs, logits

include("losses/Losses.jl")
using .Losses

include("generators/Generators.jl")
using .Generators
export GenericGenerator, GreedyGenerator, 
    generate_perturbations, conditions_satisified, mutability_constraints

include("counterfactual/Counterfactual.jl")
using .Counterfactual
export Counterfactual, initialize!, update!

include("generate_counterfactual.jl")
export generate_counterfactual

end