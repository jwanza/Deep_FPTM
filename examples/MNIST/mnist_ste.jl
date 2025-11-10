include("../../src/FuzzyPatternTM.jl")

using .FuzzyPatternTM: TMInput, TMClassifierSTE, initialize_ste!, train_ste!, discretize!, predict, accuracy, booleanize, save, compile
using MLDatasets: MNIST
using Base.Threads

# Load MNIST
train_x, train_y = MNIST.traindata()
test_x, test_y = MNIST.testdata()

# Full 10-class classification
y_train = collect(train_y)
y_test = collect(test_y)
labels = sort(unique(y_train))

# Booleanize pixels with a single threshold
function to_tm_inputs(imgs)
    X = Vector{TMInput}(undef, size(imgs, 3))
    @threads for i in 1:size(imgs, 3)
        x = Float32.(vec(imgs[:, :, i])) ./ 255f0
        X[i] = booleanize(x, 0.5f0)
    end
    return X
end

x_train = to_tm_inputs(train_x)
x_test = to_tm_inputs(test_x)

# Hyperparameters
CLAUSES = 100
L = 32
LF = 16
EPOCHS = 10
LR = 5f-3

# Train STE TM with epoch-level metrics
println("Running STE TM on MNIST (multi-class)")
println("Clauses: $(CLAUSES), L: $(L), LF: $(LF), epochs: $(EPOCHS), lr: $(LR)")
println("Classes: $(length(labels)) | Training samples: $(length(x_train)) | Test samples: $(length(x_test))")

tm = TMClassifierSTE{Int}(CLAUSES, L=L, LF=LF, tau=0.5f0)
initialize_ste!(tm, x_train, y_train)
metrics = Dict{Symbol, Any}()
train_ste!(tm, x_train, y_train;
           epochs=EPOCHS,
           lr=LR,
           ste=true,
           ste_eval=true,
           verbose=1,
           report_train_acc=true,
           report_epoch_acc=true,
           X_val=x_test,
           Y_val=y_test,
           metrics=metrics)

# Discretize and evaluate
tmc = discretize!(tm; threshold=0.5f0)
acc = accuracy(predict(tmc, x_test), y_test)
train_acc = get(metrics, :train_accuracy, nothing)
val_acc = get(metrics, :val_accuracy, nothing)
@printf("Final STE train accuracy: %.2f%%\n", train_acc === nothing ? NaN : train_acc * 100)
@printf("Final STE validation accuracy: %.2f%%\n", val_acc === nothing ? acc * 100 : val_acc * 100)
@printf("MNIST discretized compiled TM accuracy: %.2f%%\n", acc * 100)

# Save model
save(tmc, "/tmp/mnist_tm_ste_compiled.tm")


