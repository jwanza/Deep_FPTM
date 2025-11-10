using JSON3

"""
    save_compiled_to_json(tm::AbstractTMClassifier, filepath::AbstractString)

Export a compiled or trained TM into a JSON file with included literals per class.
Schema:
{
  "clauses_num": Int,
  "L": Int,
  "LF": Int,
  "classes": [
    {
      "class": String,
      "positive": [[Int...], ...],
      "negative": [[Int...], ...],
      "positive_inv": [[Int...], ...],
      "negative_inv": [[Int...], ...]
    }, ...
  ]
}
"""
function save_compiled_to_json(tm::AbstractTMClassifier, filepath::AbstractString)
    data = Dict{String, Any}()
    data["clauses_num"] = tm.clauses_num
    data["L"] = tm.L
    data["LF"] = tm.LF
    classes = Vector{Dict{String, Any}}()
    for (cls, ta) in tm.clauses
        entry = Dict{String, Any}()
        entry["class"] = string(cls)
        # Determine if team is compiled or train-time
        has_included = hasproperty(ta, :positive_included_literals)
        if has_included
            positive = [[Int(i) for i in c] for c in ta.positive_included_literals]
            negative = [[Int(i) for i in c] for c in ta.negative_included_literals]
            positive_inv = [[Int(i) for i in c] for c in ta.positive_included_literals_inverted]
            negative_inv = [[Int(i) for i in c] for c in ta.negative_included_literals_inverted]
        else
            error("Provided TM 'clauses' do not contain included literal caches. Compile or train first.")
        end
        entry["positive"] = positive
        entry["negative"] = negative
        entry["positive_inv"] = positive_inv
        entry["negative_inv"] = negative_inv
        push!(classes, entry)
    end
    data["classes"] = classes
    if !endswith(filepath, ".json")
        filepath = string(filepath, ".json")
    end
    open(filepath, "w") do io
        JSON3.write(io, data)
    end
    return filepath
end

"""
    load_compiled_from_json(filepath::AbstractString)::TMClassifierCompiled{String}

Load a compiled TM classifier from a JSON file exported with `save_compiled_to_json`
or equivalent Python exporter. Class labels are loaded as `String`.
"""
function load_compiled_from_json(filepath::AbstractString)
    data = JSON3.read(read(filepath, String))
    clauses_num = Int(data["clauses_num"])
    L = Int(data["L"])
    LF = Int(data["LF"])
    tmc = TMClassifierCompiled{String}(clauses_num, 0, 0, 0, L, LF)
    for cls_entry in data["classes"]
        cls = String(cls_entry["class"])
        tmc.clauses[cls] = TATeamCompiled(clauses_num)
        # JSON arrays are 1-based indices (Julia friendly)
        tmc.clauses[cls].positive_included_literals = [UInt16.(Vector{Int}(v)) for v in cls_entry["positive"]]
        tmc.clauses[cls].negative_included_literals = [UInt16.(Vector{Int}(v)) for v in cls_entry["negative"]]
        tmc.clauses[cls].positive_included_literals_inverted = [UInt16.(Vector{Int}(v)) for v in cls_entry["positive_inv"]]
        tmc.clauses[cls].negative_included_literals_inverted = [UInt16.(Vector{Int}(v)) for v in cls_entry["negative_inv"]]
    end
    return tmc
end


