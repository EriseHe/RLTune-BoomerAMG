# truncated_normal.jl
# Faithful Julia translation of utils/truncated_normal.m

"""
    truncated_normal(n) -> Vector{Float64}

Samples a radially-truncated standard Gaussian using rejection sampling.
Returns a vector of length n with norm <= sqrt(n).
"""
function truncated_normal(n)
    while true
        sample = randn(n)
        if norm(sample) <= sqrt(n)
            return sample
        end
    end
end
