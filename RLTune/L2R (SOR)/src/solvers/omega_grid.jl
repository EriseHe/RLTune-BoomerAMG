# Returns a grid of evenly spaced omegas plus the optimal omega for SOR
# Direct translation of omega_grid.m

function omega_grid(A, omega_min::Float64, omega_max::Float64, step::Float64)
    opt = omega_opt(A)
    before = floor(Int, (opt - omega_min) / step) + 1
    after = floor(Int, (omega_max - opt) / step) + 1
    
    grid = zeros(before + 1 + after)
    omega = omega_min
    
    for i in 1:before
        grid[i] = omega
        omega += step
    end
    
    grid[before + 1] = opt
    
    for i in (before + 2):(before + after + 1)
        grid[i] = omega
        omega += step
    end
    
    return grid
end

