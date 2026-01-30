# ChebCB.jl
# Faithful Julia translation of learners/ChebCB.m

"""
Implements the ChebCB contextual bandit algorithm for continuous
one-dimensional contexts.
NOTE: uses a time-varying setting of eta that increases linearly in t
"""
mutable struct ChebCB
    grid::Vector{Float64}
    T::Int
    d::Int
    m::Int
    t::Int
    eta::Float64
    actions::Vector{Int}
    theta::Matrix{Float64}
    features::Matrix{Float64}
    losses::Vector{Float64}
    a::Float64
    bma::Float64
    scale::Float64
    predicted::Vector{Float64}
    contexts::Vector{Float64}
end

"""
    ChebCB(grid, T, m, a, b)

# Arguments
- `grid`: action space
- `T`: number of rounds (can be set to zero if not known)
- `m`: largest power of the Chebyshev approximation
- `a`: lower bound on the context space
- `b`: upper bound on the context space
"""
function ChebCB(grid::Vector{Float64}, T::Int, m::Int, a::Float64, b::Float64)
    d = length(grid)
    return ChebCB(
        grid,
        T,
        d,
        m,
        1,
        0.0,
        zeros(Int, T),
        zeros(d, m),
        zeros(T, m),
        zeros(T),
        a,
        b - a,
        0.0,
        zeros(T),
        zeros(T)
    )
end

"""
    chebT(obj::ChebCB, x) -> Vector{Float64}

Computes Chebyshev polynomial features T_0(x), ..., T_{m-1}(x).
"""
function chebT(obj::ChebCB, x::Float64)
    out = ones(obj.m)
    if obj.m >= 2
        out[2] = x
    end
    for j in 3:obj.m
        out[j] = 2.0 * x * out[j-1] - out[j-2]
    end
    return out
end

"""
    predict!(obj::ChebCB, context) -> Float64

Predicts which action should be taken given a context as input.
"""
function predict!(obj::ChebCB, context::Float64)
    feature = chebT(obj, 2.0 / obj.bma * (context - obj.a) - 1.0)
    obj.features[obj.t, :] = feature
    yhat = obj.theta * feature
    ystar, istar = findmin(yhat)
    
    probs = zeros(obj.d)
    for i in 1:obj.d
        if i != istar
            probs[i] = 1.0 / (obj.d + obj.eta * (yhat[i] - ystar))
        end
    end
    probs[istar] = 1.0 - sum(probs)
    
    # Ensure valid probabilities
    probs = max.(probs, 0.0)
    if sum(probs) <= 0
        probs .= 1.0 / obj.d
    else
        probs ./= sum(probs)
    end
    
    i = sample(1:obj.d, Weights(probs))
    obj.actions[obj.t] = i
    
    obj.predicted[obj.t] = obj.theta[i, :]' * feature
    obj.contexts[obj.t] = context
    
    return obj.grid[i]
end

"""
    update!(obj::ChebCB, loss)

Updates the algorithm using the incurred cost.
"""
function update!(obj::ChebCB, loss)
    obj.losses[obj.t] = loss
    
    K = maximum(obj.losses[1:obj.t])
    L = (maximum(obj.losses[1:obj.t]) - minimum(obj.losses[1:obj.t])) / obj.bma
    if L <= 0
        L = 1.0
    end
    N = 2.0 + 4.0 * obj.bma * L / K * (1.0 + log(obj.m))
    should_update = obj.scale != K * N
    obj.scale = K * N
    
    ub = vcat([1.0 / N], [2.0 * obj.bma * L / obj.scale / j for j in 1:(obj.m-1)])
    lb = -ub
    
    resnorm = 0.0
    for i in 1:obj.d
        if should_update || i == obj.actions[obj.t]
            idx = findall(obj.actions[1:obj.t] .== i)
            if length(idx) > 0
                X = obj.features[idx, :]
                y = (obj.losses[idx] .- 1.0) ./ obj.scale
                
                # lsqminnorm equivalent
                obj.theta[i, :] = pinv(X) * y
                
                # Apply bounds if violated (simplified lsqlin)
                if any(abs.(obj.theta[i, :]) .> ub)
                    for j in 1:obj.m
                        obj.theta[i, j] = clamp(obj.theta[i, j], lb[j], ub[j])
                    end
                    pred = X * obj.theta[i, :]
                    resnorm = sum((y - pred).^2)
                end
            end
        end
    end
    
    alpha = (π + 2.0/π * log(2.0 * obj.m + 1)) / (2.0 * obj.scale * (obj.m + 1)) * obj.bma * L
    R = sum((obj.predicted[1:obj.t] .- obj.losses[1:obj.t] ./ K ./ N).^2)
    
    denom = R - resnorm + 2.0 * alpha^2 * obj.t
    if denom > 0
        obj.eta = 2.0 * obj.t * sqrt(obj.d * obj.t / denom)
    end
    
    obj.t = obj.t + 1
end

