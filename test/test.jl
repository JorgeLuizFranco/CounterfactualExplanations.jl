using CounterfactualExplanations
using TaijaData
using Plots
using TaijaPlotting

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
plot(ce)
