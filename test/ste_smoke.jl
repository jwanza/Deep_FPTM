include("../src/FuzzyPatternTM.jl")

using .FuzzyPatternTM: TMInput, TMClassifierSTE, initialize_ste!, train_ste!, discretize!, predict, accuracy, save_compiled_to_json, load_compiled_from_json
using Random

# Synthetic dataset
Random.seed!(123)
N = 64
D = 64
X = [TMInput(rand(Bool, D)) for _ in 1:N]
Y = [rand(Bool) for _ in 1:N]

# Train STE for a couple of epochs
tm = TMClassifierSTE{Bool}(50, L=16, LF=8, tau=0.5f0)
initialize_ste!(tm, X, Y)
train_ste!(tm, X, Y, epochs=2, lr=1f-2, ste=true, verbose=0)

# Discretize and evaluate shape/accuracy constraints
tmc = discretize!(tm; threshold=0.5f0)
pred = predict(tmc, X)
acc = accuracy(pred, Y)
@assert length(pred) == length(Y)
@assert 0.0 <= acc <= 1.0

# JSON bridge: export and import back, predictions must match
json_path = save_compiled_to_json(tmc, "/tmp/ste_tm.json")
tmc2 = load_compiled_from_json(json_path)
pred2 = predict(tmc2, X)
@assert length(pred2) == length(Y)
acc2 = accuracy(pred2, [string(y) for y in Y])
@assert 0.0 <= acc2 <= 1.0

println("Julia STE smoke test passed. Acc1 = ", acc, " Acc2 = ", acc2)


