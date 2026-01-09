# laplacian.jl
# Creates 2D discrete Laplacian matching MATLAB's delsq(numgrid('S', n))

"""
    create_laplacian_2d(n) -> SparseMatrixCSC

Creates the 2D discrete Laplacian on an (n-2)×(n-2) interior grid.
Equivalent to MATLAB's delsq(numgrid('S', n)).
"""
function create_laplacian_2d(n)
    m = n - 2  # interior grid size
    N = m * m
    
    I_idx = Int[]
    J_idx = Int[]
    V = Float64[]
    
    for i in 1:m
        for j in 1:m
            idx = (i - 1) * m + j
            
            # Diagonal: 4
            push!(I_idx, idx); push!(J_idx, idx); push!(V, 4.0)
            
            # Left neighbor
            if j > 1
                push!(I_idx, idx); push!(J_idx, idx - 1); push!(V, -1.0)
            end
            
            # Right neighbor
            if j < m
                push!(I_idx, idx); push!(J_idx, idx + 1); push!(V, -1.0)
            end
            
            # Bottom neighbor
            if i > 1
                push!(I_idx, idx); push!(J_idx, idx - m); push!(V, -1.0)
            end
            
            # Top neighbor
            if i < m
                push!(I_idx, idx); push!(J_idx, idx + m); push!(V, -1.0)
            end
        end
    end
    
    return sparse(I_idx, J_idx, V, N, N)
end
