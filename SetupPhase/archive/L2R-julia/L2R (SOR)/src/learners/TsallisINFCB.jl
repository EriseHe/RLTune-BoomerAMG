# TsallisINFCB.jl
# Faithful Julia translation of learners/TsallisINFCB.m

"""
Implements a contextual bandit algorithm that discretizes the context
space and runs Tsallis-INF independently in each bin.
"""
mutable struct TsallisINFCB
    grid::Vector{TsallisINF}
    disc::Vector{Float64}
    action::Int
end

"""
    TsallisINFCB(grid, m, a, b)

# Arguments
- `grid`: action space
- `m`: number of bins for discretizing the context space
- `a`: lower bound on the context space
- `b`: upper bound on the context space
"""
function TsallisINFCB(grid::Vector{Float64}, m::Int, a::Float64, b::Float64)
    handles = [TsallisINF(grid, 0) for _ in 1:m]
    disc = a .+ (b - a) .* range(0.5/m, 1.0 - 0.5/m, length=m)
    return TsallisINFCB(handles, collect(disc), 1)
end

"""
    predict!(obj::TsallisINFCB, context) -> Float64

Predicts which action should be taken given a context as input.
"""
function predict!(obj::TsallisINFCB, context::Float64)
    _, i = findmin(abs.(obj.disc .- context))
    out = predict!(obj.grid[i])
    obj.action = i
    return out
end

"""
    update!(obj::TsallisINFCB, loss)

Updates the algorithm using the incurred cost.
"""
function update!(obj::TsallisINFCB, loss)
    update!(obj.grid[obj.action], loss)
end

