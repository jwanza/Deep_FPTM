# Differentiable TM with Straight-Through Estimation (STE)
#
# This file extends the Fuzzy-Pattern TM with an optional differentiable path.
# It introduces continuous include probabilities for literals and evaluates
# clauses with a product t-norm. An optional STE hardening mirrors discrete
# behavior in the forward pass while keeping gradients in the backward pass.
#
# Notes:
# - Training uses Flux/Zygote AD without multithreading to avoid nested parallelism.
# - Discretization converts probabilities into a compiled TM compatible with
#   the existing inference path and serialization.

using Flux
using Zygote

############################
# Trainable STE structures #
############################

mutable struct ContinuousTeamSTE
    positive::Matrix{Float32}
    negative::Matrix{Float32}
    positive_inv::Matrix{Float32}
    negative_inv::Matrix{Float32}
    clause_size::Int64
end

mutable struct TMClassifierSTE{ClassType}
    clauses_num::Int64
    L::Int64
    LF::Int64
    tau::Float32
    clauses::Dict{ClassType, ContinuousTeamSTE}
end

Flux.@functor ContinuousTeamSTE
Flux.trainable(m::ContinuousTeamSTE) = (m.positive, m.negative, m.positive_inv, m.negative_inv)

Flux.@functor TMClassifierSTE
Flux.trainable(m::TMClassifierSTE) = (collect(values(m.clauses)),)

#####################################
# Initialization and helper methods #
#####################################

function class_keys(tm::TMClassifierSTE)
    return collect(keys(tm.clauses))
end
Zygote.@nograd class_keys

function TMClassifierSTE{ClassType}(clauses_num::Int64; L::Int64=16, LF::Int64=4, tau::Float32=0.5) where {ClassType}
    return TMClassifierSTE{ClassType}(clauses_num, L, LF, tau, Dict{ClassType, ContinuousTeamSTE}())
end

function initialize_ste!(tm::TMClassifierSTE, X::Vector{TMInput}, Y::Vector)
    clause_size = length(first(X))
    half_clauses = floor(Int, tm.clauses_num / 2)
    for cls in unique(Y)
        tm.clauses[cls] = ContinuousTeamSTE(
            rand(Float32, clause_size, half_clauses) .* 0.05f0,
            rand(Float32, clause_size, half_clauses) .* 0.05f0,
            rand(Float32, clause_size, half_clauses) .* 0.05f0,
            rand(Float32, clause_size, half_clauses) .* 0.05f0,
            clause_size
        )
    end
    return tm
end

#############################
# Differentiable evaluation #
#############################

@inline function _ste_mask(p::AbstractArray{<:Real}, tau::Real)
    ph = Float32.(p .> tau)
    return ph .+ (Float32.(p) .- ph)
end

@inline function _literal_factor(pos_prob::AbstractArray{<:Real}, inv_prob::AbstractArray{<:Real}, xbit::Float32, ste::Bool, tau::Float32)::Tuple{Array{Float32}, Array{Float32}}
    if ste
        p = _ste_mask(pos_prob, tau)
        q = _ste_mask(inv_prob, tau)
        return (1f0 .- p .+ p .* xbit), (1f0 .- q .+ q .* (1f0 - xbit))
    else
        p = Float32.(pos_prob)
        q = Float32.(inv_prob)
        return (1f0 .- p .+ p .* xbit), (1f0 .- q .+ q .* (1f0 - xbit))
    end
end

function clause_vote_ste(ta::ContinuousTeamSTE, x::TMInput; ste::Bool=true, tau::Float32=0.5)::Tuple{Float32, Float32}
    # Vectorised clause evaluation using log-linear penalty (matches PyTorch STE implementation).
    xbits = Float32.(x.x)
    xneg = 1f0 .- xbits

    p_pos = ste ? _ste_mask(ta.positive, tau) : Float32.(ta.positive)
    p_neg = ste ? _ste_mask(ta.negative, tau) : Float32.(ta.negative)
    p_pos_inv = ste ? _ste_mask(ta.positive_inv, tau) : Float32.(ta.positive_inv)
    p_neg_inv = ste ? _ste_mask(ta.negative_inv, tau) : Float32.(ta.negative_inv)

    scale = 4f0 / ta.clause_size
    pos_score = (p_pos' * xneg) .+ (p_pos_inv' * xbits)
    neg_score = (p_neg' * xneg) .+ (p_neg_inv' * xbits)

    pos_prod = exp.(-clamp.(scale .* vec(pos_score), 0f0, 10f0))
    neg_prod = exp.(-clamp.(scale .* vec(neg_score), 0f0, 10f0))

    return (sum(pos_prod), sum(neg_prod))
end

function logits_ste(tm::TMClassifierSTE, x::TMInput; ste::Bool=true)
    # Returns Dict{ClassType, Float32} of class scores
    out = Dict{typeof(first(keys(tm.clauses))), Float32}()
    for (cls, ta) in tm.clauses
        pos, neg = clause_vote_ste(ta, x; ste=ste, tau=tm.tau)
        out[cls] = pos - neg
    end
    return out
end

function predict_proba_ste(tm::TMClassifierSTE, x::TMInput; ste::Bool=true)
    ks = class_keys(tm)
    ls = [logits_ste(tm, x; ste=ste)[k] for k in ks]
    exps = exp.(ls .- maximum(ls))
    ps = exps ./ sum(exps)
    return ks, ps
end

function predict(tm::TMClassifierSTE, x::TMInput; ste::Bool=true)
    ks, ps = predict_proba_ste(tm, x; ste=ste)
    return ks[argmax(ps)]
end

function predict(tm::TMClassifierSTE, X::Vector{TMInput}; ste::Bool=true)
    predicted = Vector{eltype(collect(keys(tm.clauses)))}(undef, length(X))
    @inbounds for i in eachindex(X)
        predicted[i] = predict(tm, X[i]; ste=ste)
    end
    return predicted
end

##########################
# STE training utilities #
##########################

function _cross_entropy(ps::AbstractVector{<:Real}, yidx::Int)
    return -log(max(eps(Float32), Float32(ps[yidx])))
end

function _loss_dataset(tm::TMClassifierSTE, X::Vector{TMInput}, Y::Vector; ste::Bool=true)
    # Map labels to index order
    classes = class_keys(tm)
    idx_of = Dict(c => i for (i, c) in enumerate(classes))
    total::Float32 = 0f0
    @inbounds for (x, y) in zip(X, Y)
        ks, ps = predict_proba_ste(tm, x; ste=ste)
        total += _cross_entropy(ps, idx_of[y])
    end
    return total / length(Y)
end

"""
    train_ste!(tm, X, Y; epochs=10, lr=1e-3, ste=true, verbose=1)

Train a differentiable TM with STE using Flux/Zygote.
"""
function train_ste!(tm::TMClassifierSTE, X::Vector{TMInput}, Y::Vector;
                    epochs::Int=10, lr::Float32=1f-3, ste::Bool=true, ste_eval::Bool=true,
                    verbose::Int=1, report_train_acc::Bool=true, report_epoch_acc::Bool=false,
                    X_val::Union{Nothing,Vector{TMInput}}=nothing, Y_val::Union{Nothing,Vector}=nothing,
                    metrics::Union{Nothing,Dict{Symbol,Any}}=nothing)
    if isempty(tm.clauses)
        initialize_ste!(tm, X, Y)
    end
    ps = Flux.params(tm)
    opt = Flux.Optimise.Descent(lr)
    train_hist = report_epoch_acc ? Float32[] : nothing
    val_hist = (report_epoch_acc && X_val !== nothing && Y_val !== nothing) ? Float32[] : nothing
    last_train_acc = nothing
    last_val_acc = nothing
    t_start = time()
    for e in 1:epochs
        loss_val, back = Zygote.pullback(() -> _loss_dataset(tm, X, Y; ste=ste), ps)
        gs = back(1f0)
        Flux.Optimise.update!(opt, ps, gs)
        epoch_acc = nothing
        val_acc = nothing
        if report_train_acc
            preds = predict(tm, X; ste=ste_eval)
            epoch_acc = Float32(accuracy(preds, Y))
            last_train_acc = epoch_acc
            if report_epoch_acc && train_hist !== nothing
                push!(train_hist, epoch_acc)
            end
        end
        if report_epoch_acc && X_val !== nothing && Y_val !== nothing
            preds_val = predict(tm, X_val; ste=ste_eval)
            val_acc = Float32(accuracy(preds_val, Y_val))
            last_val_acc = val_acc
            if val_hist !== nothing
                push!(val_hist, val_acc)
            end
        end
        if verbose > 0
            if report_epoch_acc && epoch_acc !== nothing
                if val_acc !== nothing
                    @printf("#%d  STE loss: %.6f  train_acc: %.4f  val_acc: %.4f\n", e, loss_val, epoch_acc, val_acc)
                else
                    @printf("#%d  STE loss: %.6f  train_acc: %.4f\n", e, loss_val, epoch_acc)
                end
            else
                @printf("#%d  STE loss: %.6f\n", e, loss_val)
            end
        end
    end
    total_time = time() - t_start
    final_train = last_train_acc !== nothing ? last_train_acc : Float32(accuracy(predict(tm, X; ste=ste_eval), Y))
    final_val = nothing
    if X_val !== nothing && Y_val !== nothing
        final_val = Float32(accuracy(predict(tm, X_val; ste=ste_eval), Y_val))
    end
    if metrics !== nothing
        metrics[:train_accuracy] = final_train
        metrics[:val_accuracy] = final_val
        metrics[:train_history] = train_hist
        metrics[:val_history] = val_hist
        metrics[:train_time_s] = total_time
    end
    return tm
end

#############################
# Discretization to compiled #
#############################

"""
    discretize!(tm; threshold=0.5)

Convert continuous include probabilities into a compiled TM compatible with the
existing inference path. Returns `TMClassifierCompiled`.
"""
function discretize!(tm::TMClassifierSTE; threshold::Float32=0.5)
    # Build a compiled classifier with included literals per clause
    tmc = TMClassifierCompiled{eltype(collect(keys(tm.clauses)))}(tm.clauses_num, 0, 0, 0, tm.L, tm.LF)
    for (cls, ta) in tm.clauses
        tmc.clauses[cls] = TATeamCompiled(tm.clauses_num)
        half = size(ta.positive, 2)
        # Positive
        for j in 1:half
            pos_idxs = Vector{UInt16}()
            posinv_idxs = Vector{UInt16}()
            @inbounds for i in 1:ta.clause_size
                if ta.positive[i, j] >= threshold
                    push!(pos_idxs, UInt16(i))
                end
                if ta.positive_inv[i, j] >= threshold
                    push!(posinv_idxs, UInt16(i))
                end
            end
            tmc.clauses[cls].positive_included_literals[j] = pos_idxs
            tmc.clauses[cls].positive_included_literals_inverted[j] = posinv_idxs
        end
        # Negative
        for j in 1:half
            neg_idxs = Vector{UInt16}()
            neginv_idxs = Vector{UInt16}()
            @inbounds for i in 1:ta.clause_size
                if ta.negative[i, j] >= threshold
                    push!(neg_idxs, UInt16(i))
                end
                if ta.negative_inv[i, j] >= threshold
                    push!(neginv_idxs, UInt16(i))
                end
            end
            tmc.clauses[cls].negative_included_literals[j] = neg_idxs
            tmc.clauses[cls].negative_included_literals_inverted[j] = neginv_idxs
        end
    end
    return tmc
end


