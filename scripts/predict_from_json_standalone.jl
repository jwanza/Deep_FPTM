using JSON3

struct CompiledTeam
    pos::Vector{Vector{Int}}
    neg::Vector{Vector{Int}}
    pos_inv::Vector{Vector{Int}}
    neg_inv::Vector{Vector{Int}}
end

struct CompiledModel
    clauses_num::Int
    L::Int
    LF::Int
    classes::Vector{String}
    teams::Dict{String,CompiledTeam}
end

function load_compiled_json(path::AbstractString)::CompiledModel
    data = JSON3.read(read(path, String))
    clauses_num = Int(data["clauses_num"])
    L = Int(data["L"])
    LF = Int(data["LF"])
    classes = String[]
    teams = Dict{String,CompiledTeam}()
    for entry in data["classes"]
        cls = String(entry["class"])
        push!(classes, cls)
        teams[cls] = CompiledTeam(
            [Vector{Int}(v) for v in entry["positive"]],
            [Vector{Int}(v) for v in entry["negative"]],
            [Vector{Int}(v) for v in entry["positive_inv"]],
            [Vector{Int}(v) for v in entry["negative_inv"]],
        )
    end
    return CompiledModel(clauses_num, L, LF, classes, teams)
end

struct TMInputLocal
    x::Vector{Bool}
end

function read_bool_dataset(path::AbstractString)
    lines = readlines(path)
    X = Vector{TMInputLocal}(undef, length(lines))
    Y = Vector{String}(undef, length(lines))
    for (i, ln) in enumerate(lines)
        parts = split(ln, " ")
        bits = [parse(Int, x) != 0 for x in parts[1:end-1]]
        X[i] = TMInputLocal(bits)
        Y[i] = parts[end]
    end
    return X, Y
end

function check_clause(x::TMInputLocal, literals::Vector{Int}, literals_inv::Vector{Int}, LF::Int)::Int
    c = min(LF, length(literals) + length(literals_inv))
    for idx in literals
        c <= 0 && return 0
        c -= (!x.x[idx])
    end
    for idx in literals_inv
        c <= 0 && return 0
        c -= (x.x[idx])
    end
    return c
end

function vote(team::CompiledTeam, x::TMInputLocal, LF::Int)::Tuple{Int,Int}
    pos = 0
    neg = 0
    for j in eachindex(team.pos)
        pos += check_clause(x, team.pos[j], team.pos_inv[j], LF)
    end
    for j in eachindex(team.neg)
        neg += check_clause(x, team.neg[j], team.neg_inv[j], LF)
    end
    return pos, neg
end

function predict(model::CompiledModel, x::TMInputLocal)::String
    best = -typemax(Int)
    best_cls = model.classes[1]
    for cls in model.classes
        vpos, vneg = vote(model.teams[cls], x, model.LF)
        v = vpos - vneg
        if v > best
            best = v
            best_cls = cls
        end
    end
    return best_cls
end

function accuracy(preds::Vector{String}, Y::Vector{String})
    @assert length(preds) == length(Y)
    c = 0
    for i in eachindex(Y)
        c += (preds[i] == Y[i])
    end
    return c / length(Y)
end

function main(json_path::AbstractString, test_path::AbstractString)
    model = load_compiled_json(json_path)
    X, Y = read_bool_dataset(test_path)
    preds = Vector{String}(undef, length(X))
    for i in eachindex(X)
        preds[i] = predict(model, X[i])
    end
    println("Samples: ", length(Y))
    println("Accuracy: ", round(accuracy(preds, Y)*100; digits=2), "%")
end

if abspath(PROGRAM_FILE) == @__FILE__
    if length(ARGS) < 2
        println("Usage: julia scripts/predict_from_json_standalone.jl /tmp/tm_mnist.json /tmp/MNISTTestData.txt")
    else
        main(ARGS[1], ARGS[2])
    end
end


