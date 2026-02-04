# TsallisINF.jl
# Faithful Julia translation of learners/TsallisINF.m

"""
Implements the Tsallis-INF bandit algorithm.
NOTE: uses a time-varying setting of eta = 2 / sqrt(t)
"""
mutable struct TsallisINF
    grid::Vector{Float64}
    d::Int
    t::Int
    k::Vector{Float64}
    index::Int
    prob::Float64
    scale::Float64
    actions::Vector{Int}
    losses::Vector{Float64}
end

"""
    TsallisINF(grid, T)

# Arguments
- `grid`: action space
- `T`: number of rounds (can be set to zero if not known)
"""
function TsallisINF(grid::Vector{Float64}, T::Int)
    d = length(grid)
    return TsallisINF(
        grid,
        d,
        1,
        zeros(d),
        0,
        1.0 / d,
        1.0,
        zeros(Int, max(T, 1)),
        zeros(max(T, 1))
    )
end

"""
    predict!(obj::TsallisINF) -> Float64

Samples action to be taken.
"""
function predict!(obj::TsallisINF)
    eta = 2.0 / sqrt(obj.t)
    x = -1.0
    probs = zeros(obj.d)
    
    for _ in 1:20
        probs .= 4.0 .* (eta .* (obj.k ./ obj.scale .- x)) .^ (-2.0)
        x = x - (sum(probs) - 1.0) / (eta * sum(probs .^ 1.5))
    end
    
    # Sample action
    try
        obj.index = sample(1:obj.d, Weights(probs))
    catch
        obj.index = rand(1:obj.d)
        obj.prob = 1.0 / obj.d
    end
    obj.prob = probs[obj.index]
    
    # Expand arrays if needed
    if obj.t > length(obj.actions)
        append!(obj.actions, zeros(Int, 1000))
        append!(obj.losses, zeros(1000))
    end
    obj.actions[obj.t] = obj.index
    
    return obj.grid[obj.index]
end

"""
    update!(obj::TsallisINF, loss)

Updates action distribution using the incurred cost.
"""
function update!(obj::TsallisINF, loss)
    obj.losses[obj.t] = loss
    obj.scale = mean(obj.losses[1:obj.t]) - 1.0
    obj.k[obj.index] = obj.k[obj.index] + (loss - 1.0) / obj.prob
    obj.t = obj.t + 1
end

