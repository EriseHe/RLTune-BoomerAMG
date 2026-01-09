# visualization.jl
# Supplementary visualization for bandit parameter selection
# NOT part of original MATLAB translation

"""
    generate_bandit_visualizations(chosen_params, action_probs, costs, param_grid, param_label, output_dir, title_prefix)

Generate visualization of bandit parameter choices over instances.
"""
function generate_bandit_visualizations(
    chosen_params::Vector{Float64},
    action_probs::Matrix{Float64},
    costs::Vector{Float64},
    param_grid::Vector{Float64},
    param_label::String,
    output_dir::String,
    title_prefix::String
)
    T = length(chosen_params)
    mkpath(output_dir)
    
    println("\nGenerating bandit visualization...")
    
    # Bandit parameter choices over instances
    p = plot(size=(900, 400), dpi=150, legend=:topright)
    scatter!(p, 1:T, chosen_params, alpha=0.2, markersize=2,
             color=:blue, label="Chosen $param_label", markerstrokewidth=0)
    
    # Running average
    window = 100
    running_avg = [mean(chosen_params[max(1,i-window+1):i]) for i in 1:T]
    plot!(p, 1:T, running_avg, linewidth=2, color=:red,
          label="Running avg (window=$window)")
    
    xlabel!(p, "Instance")
    ylabel!(p, "$param_label")
    title!(p, "$title_prefix: Bandit Parameter Selection Over Instances")
    
    savefig(p, joinpath(output_dir, "bandit_choices_over_instances.png"))
    println("Saved bandit_choices_over_instances.png")
end
