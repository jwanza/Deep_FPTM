include("../src/FuzzyPatternTM.jl")
include("../src/JsonBridge.jl")

using .FuzzyPatternTM: TMInput, predict, accuracy, load_compiled_from_json

function read_bool_dataset(path::AbstractString)
    lines = readlines(path)
    X = Vector{TMInput}(undef, length(lines))
    Y = Vector{String}(undef, length(lines))
    for (i, ln) in enumerate(lines)
        parts = split(ln, " ")
        bits = [parse(Bool, x) for x in parts[1:end-1]]
        X[i] = TMInput(bits)
        Y[i] = parts[end] # keep as string label
    end
    return X, Y
end

function main(json_path::AbstractString, test_path::AbstractString)
    tm = load_compiled_from_json(json_path)
    X, Y = read_bool_dataset(test_path)
    preds = predict(tm, X)
    # Normalize prediction types to String
    preds_s = [string(p) for p in preds]
    acc = accuracy(preds_s, Y)
    println("JSON: ", json_path)
    println("Samples: ", length(Y))
    println("Accuracy: ", round(acc*100; digits=2), "%")
end

if abspath(PROGRAM_FILE) == @__FILE__
    if length(ARGS) < 2
        println("Usage: julia scripts/predict_from_json.jl /tmp/tm_mnist.json /tmp/MNISTTestData.txt")
    else
        main(ARGS[1], ARGS[2])
    end
end


