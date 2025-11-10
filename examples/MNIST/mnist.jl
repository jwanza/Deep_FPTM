include("../../src/FuzzyPatternTM.jl")

try
    using MLDatasets: MNIST, FashionMNIST
catch LoadError
    import Pkg
    Pkg.add("MLDatasets")
end

using Printf: @printf
using MLDatasets: MNIST, FashionMNIST
using .FuzzyPatternTM: TMInput, TMClassifier, train!, predict, accuracy, save, load, booleanize, combine, optimize!, benchmark

if !haskey(ENV, "DATADEPS_ALWAYS_ACCEPT")
    ENV["DATADEPS_ALWAYS_ACCEPT"] = "true"
end

train_imgs, train_labels = MNIST.traindata(Float32)
test_imgs, test_labels = MNIST.testdata(Float32)
# To switch to Fashion-MNIST instead, replace the two lines above with:
# train_imgs, train_labels = FashionMNIST.traindata(Float32)
# test_imgs, test_labels = FashionMNIST.testdata(Float32)

x_train = [view(train_imgs, :, :, i) for i in axes(train_imgs, 3)]
x_test = [view(test_imgs, :, :, i) for i in axes(test_imgs, 3)]

# 4-bit booleanization
x_train = [booleanize(x, 0, 0.25, 0.5, 0.75) for x in x_train]
x_test = [booleanize(x, 0, 0.25, 0.5, 0.75) for x in x_test]

# Convert y_train and y_test to the Int8 type to save memory
y_train = Int8.(train_labels)
y_test = Int8.(test_labels)

CLAUSES = 20
T = 20
S = 200
L = 150
LF = 75

# CLAUSES = 200
# T = 20
# S = 200
# L = 16
# LF = 8

# CLAUSES = 512
# T = 32
# S = 200
# L = 16
# LF = 8

# CLAUSES = 2000
# T = 100
# S = 350
# L = 20
# LF = 10

# CLAUSES = 40
# T = 10
# S = 125
# L = 10
# LF = 5

EPOCHS = 2000
best_tms_size = 512

# Training the TM model
tm = TMClassifier{eltype(y_train)}(CLAUSES, T, S, L=L, LF=LF, states_num=256, include_limit=200)  # include_limit=200 instead of 128 but you can try different numbers.
tms = train!(tm, x_train, y_train, x_test, y_test, EPOCHS, best_tms_size=best_tms_size, shuffle=true, batch=true, verbose=1)

save(tms, "/tmp/tms.tm")
tms = load("/tmp/tms.tm")

# Binomial combinatorial merge of trained TM models
tm, _ = combine(tms, 2, x_test, y_test, batch=true)
save(tm, "/tmp/tm2.tm")
tm = load("/tmp/tm2.tm")

# Optimizing the TM model
optimize!(tm, x_train)
save(tm, "/tmp/tm_optimized.tm")
tm_opt = load("/tmp/tm_optimized.tm")

benchmark(tm_opt, x_test, y_test, 5000, batch=true, warmup=true)
