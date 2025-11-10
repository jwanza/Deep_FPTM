include("../../src/FuzzyPatternTM.jl")

using Base.Threads
using .FuzzyPatternTM: TMInput, TMClassifierSTE, initialize_ste!, train_ste!, discretize!, compile, predict, accuracy, save

# Loading datasets
train = readlines("/tmp/IMDBTrainingData.txt")
test = readlines("/tmp/IMDBTestData.txt")

# Preparing datasets
x_train::Vector{TMInput} = Vector{TMInput}(undef, length(train))
y_train::Vector{Bool} = Vector{Bool}(undef, length(train))
@threads for i in eachindex(train)
    xy = [parse(Bool, x) for x in split(train[i], " ")]
    x_train[i] = TMInput(xy[1:length(xy) - 1])
    y_train[i] = xy[length(xy)]
end
x_test::Vector{TMInput} = Vector{TMInput}(undef, length(test))
y_test::Vector{Bool} = Vector{Bool}(undef, length(test))
@threads for i in eachindex(test)
    xy = [parse(Bool, x) for x in split(test[i], " ")]
    x_test[i] = TMInput(xy[1:length(xy) - 1])
    y_test[i] = xy[length(xy)]
end

# Hyperparameters
CLAUSES = 200
L = 64
LF = 64
EPOCHS = 10  # STE demo (use higher for better accuracy)
LR = 1f-2

# Train STE TM
tm = TMClassifierSTE{eltype(y_train)}(CLAUSES, L=L, LF=LF, tau=0.5f0)
initialize_ste!(tm, x_train, y_train)
train_ste!(tm, x_train, y_train, epochs=EPOCHS, lr=LR, ste=true, verbose=1)

# Discretize to compiled TM and evaluate
tmc = discretize!(tm; threshold=0.5f0)
acc = accuracy(predict(tmc, x_test), y_test)
@printf("Discretized compiled TM accuracy: %.2f%%\n", acc * 100)

# Save model
save(tmc, "/tmp/tm_ste_compiled.tm")


