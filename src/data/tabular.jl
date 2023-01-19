using CounterfactualExplanations
using CSV
using DataFrames
using LazyArtifacts
using MLJBase
using MLJModels: ContinuousEncoder

data_dir = joinpath(artifact"data-tabular", "data-tabular")

"""
    load_california_housing()

Loads and prepares California Housing data.
"""
function load_california_housing()
    df = CSV.read(joinpath(data_dir, "cal_housing.csv"), DataFrame)
    X = permutedims(Matrix(df[:, Not(:target)]))
    y = df.target
    counterfactual_data = CounterfactualData(X, y)
    return counterfactual_data
end

"""
    load_gmsc()

Loads and prepares Give Me Some Credit (GMSC) data.
"""
function load_gmsc()
    df = CSV.read(joinpath(data_dir, "gmsc.csv"), DataFrame)
    X = permutedims(Matrix(df[:, Not(:target)]))
    y = df.target
    counterfactual_data = CounterfactualData(X, y)
    return counterfactual_data
end

"""
    load_credit_default()

Loads and prepares UCI Credit Default data.
"""
function load_credit_default()

    df = CSV.read(joinpath(data_dir, "credit_default.csv"), DataFrame)
    y = df.target

    # Categorical encoding:
    df.SEX = categorical(df.SEX)
    df.EDUCATION = categorical(df.EDUCATION)
    df.MARRIAGE = categorical(df.MARRIAGE)
    transformer = ContinuousEncoder()
    mach = MLJBase.fit!(machine(transformer, df[:, Not(:target)]))
    X = MLJBase.transform(mach, df[:, Not(:target)])
    X = permutedims(Matrix(X))
    features_categorical = [
        [2, 3],             # SEX
        collect(4:10),      # EDUCATION
        collect(11:14)      # MARRIAGE
    ]

    counterfactual_data = CounterfactualData(
        X, y;
        features_categorical=features_categorical
    )

    return counterfactual_data
end