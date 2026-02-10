"""
    MaterialSegments.jl

Handles multi-material bed configurations for PSA optimization.
"""

"""
    MaterialSegment

Represents a single material segment in the bed.

Fields:
- `material_idx`: Index of material from material library (can be fractional before rounding)
- `start_node`: Starting discretization node (1-based)
- `end_node`: Ending discretization node (inclusive)
- `fraction`: Bed length fraction (0-1)
- `isotherm_params`: All 13 isotherm parameters from ISOTHERM_PARAMETERS
- `ro_s`: Solid density [kg/m³]
- `C_ps`: Solid heat capacity [J/(kg·K)]
- `q_s0`: Volumetric saturation capacity [mol/m³] = q_s × ro_s
"""
struct MaterialSegment
    material_idx::Float64
    start_node::Int
    end_node::Int
    fraction::Float64
    isotherm_params::Vector{Float64}
    ro_s::Float64
    C_ps::Float64
    q_s0::Float64
    deltaU_1::Float64  # Heat of adsorption for CO2 [J/mol]
    deltaU_2::Float64  # Heat of adsorption for N2 [J/mol]
end

"""
    MultiMaterialBed

Container for multi-material bed configuration.

Fields:
- `segments`: Vector of MaterialSegment
- `N`: Total number of discretization nodes
- `is_uniform`: Flag indicating if this is a single-material bed (for performance)
"""
struct MultiMaterialBed
    segments::Vector{MaterialSegment}
    N::Int
    is_uniform::Bool
end

"""
    create_uniform_bed(N::Int, isotherm_params::Vector{Float64}, ro_s::Float64=1000.0, C_ps::Float64=1070.0)

Create a single-material bed (backward compatible with existing code).
"""
function create_uniform_bed(N::Int, isotherm_params::Vector{Float64}, ro_s::Float64=1000.0, C_ps::Float64=1070.0, deltaU_1::Float64=-30000.0, deltaU_2::Float64=-15000.0)
    q_s = 5.84  # mol/kg - specific saturation capacity (constant)
    q_s0 = q_s * ro_s  # mol/m³ - volumetric saturation capacity
    segment = MaterialSegment(1.0, 1, N, 1.0, isotherm_params, ro_s, C_ps, q_s0, deltaU_1, deltaU_2)
    return MultiMaterialBed([segment], N, true)
end

"""
    create_multi_material_bed(N::Int, 
                              material_indices::Vector{Float64},
                              fractions::Vector{Float64},
                              isotherm_library::Matrix{Float64},
                              material_props::Matrix{Float64})

Create a multi-material bed from optimization variables.

# Arguments
- `N`: Number of discretization nodes
- `material_indices`: Real-valued material indices (e.g., [2.3, 5.7, 8.1])
- `fractions`: Bed length fractions for each segment (must sum to 1.0)
- `isotherm_library`: Matrix where row i contains isotherm params [b_inf_1, b_inf_2, dH_1, dH_2, ...]
- `material_props`: Matrix where row i contains [ro_s, deltaU_1, deltaU_2] (from SIMULATION_PARAMETERS)

# Returns
- `MultiMaterialBed` object

# Example
```julia
# 3 segments with different materials
material_indices = [2.3, 5.7, 8.1]  # Will select materials 2, 6, 8
fractions = [0.3, 0.4, 0.3]  # 30%, 40%, 30% of bed length

bed = create_multi_material_bed(10, material_indices, fractions, isotherm_library, material_props)
```
"""
function create_multi_material_bed(N::Int, 
                                   material_indices::Vector{Float64},
                                   fractions::Vector{Float64},
                                   isotherm_library::Matrix{Float64},
                                   material_props::Matrix{Float64})
    
    # Validate inputs
    @assert length(material_indices) == length(fractions) "Mismatch in number of segments"
    @assert sum(fractions) ≈ 1.0 "Fractions must sum to 1.0"
    @assert all(fractions .>= 0) "All fractions must be non-negative"
    @assert size(isotherm_library, 1) == size(material_props, 1) "Isotherm and material property matrices must have same number of rows"
    
    n_segments = length(material_indices)
    segments = MaterialSegment[]
    
    # Calculate node ranges for each segment
    cumulative_fraction = 0.0
    current_node = 1
    
    for i in 1:n_segments
        # Round material index to nearest integer
        mat_idx_rounded = round(Int, material_indices[i])
        mat_idx_rounded = clamp(mat_idx_rounded, 1, size(isotherm_library, 1))
        
        # Calculate node range
        cumulative_fraction += fractions[i]
        next_node = round(Int, cumulative_fraction * N)
        next_node = min(next_node, N)  # Ensure we don't exceed N
        
        # Handle last segment to ensure it reaches N
        if i == n_segments
            next_node = N
        end
        
        # Extract isotherm parameters from library (all 13 columns)
        isotherm_params = isotherm_library[mat_idx_rounded, :]
        
        # Extract material properties
        ro_s = material_props[mat_idx_rounded, 1]  # Solid density from column 1
        deltaU_1 = material_props[mat_idx_rounded, 2]  # Heat of adsorption CO2 from column 2
        deltaU_2 = material_props[mat_idx_rounded, 3]  # Heat of adsorption N2 from column 3
        C_ps = 1070.0  # Default heat capacity (could be extended if data available)
        
        # Compute volumetric saturation capacity
        q_s = 5.84  # mol/kg - specific saturation capacity (constant)
        q_s0 = q_s * ro_s  # mol/m³ - volumetric saturation capacity
        
        # Create segment
        segment = MaterialSegment(
            material_indices[i],  # Keep original (fractional) for tracking
            current_node,
            next_node,
            fractions[i],
            isotherm_params,
            ro_s,
            C_ps,
            q_s0,
            deltaU_1,
            deltaU_2
        )
        
        push!(segments, segment)
        current_node = next_node + 1
    end
    
    # Check if effectively uniform (all same material)
    is_uniform = length(unique([round(Int, s.material_idx) for s in segments])) == 1
    
    return MultiMaterialBed(segments, N, is_uniform)
end

"""
    get_isotherm_params(bed::MultiMaterialBed, node::Int)

Get isotherm parameters for a specific node.

# Arguments
- `bed`: MultiMaterialBed object
- `node`: Node index (1-based)

# Returns
- Vector of isotherm parameters [b_inf_1, b_inf_2, dH_1, dH_2]
"""
function get_isotherm_params(bed::MultiMaterialBed, node::Int)
    for segment in bed.segments
        if segment.start_node <= node <= segment.end_node
            return segment.isotherm_params
        end
    end
    error("Node $node not found in any segment")
end

"""
    get_segment(bed::MultiMaterialBed, node::Int)

Get the segment containing a specific node.
"""
function get_segment(bed::MultiMaterialBed, node::Int)
    for segment in bed.segments
        if segment.start_node <= node <= segment.end_node
            return segment
        end
    end
    error("Node $node not found in any segment")
end

"""
    get_material_properties(bed::MultiMaterialBed, node::Int)

Get solid density, heat capacity, saturation capacity, and heats of adsorption for a specific node.

# Arguments
- `bed`: MultiMaterialBed object
- `node`: Node index (1-based)

# Returns
- Tuple: (ro_s, C_ps, q_s0, deltaU_1, deltaU_2) where:
  - ro_s: Solid density [kg/m³]
  - C_ps: Solid heat capacity [J/(kg·K)]
  - q_s0: Volumetric saturation capacity [mol/m³]
  - deltaU_1: Heat of adsorption for CO2 [J/mol]
  - deltaU_2: Heat of adsorption for N2 [J/mol]
"""
function get_material_properties(bed::MultiMaterialBed, node::Int)
    segment = get_segment(bed, node)
    return (segment.ro_s, segment.C_ps, segment.q_s0, segment.deltaU_1, segment.deltaU_2)
end

"""
    optimize_decision_vars_to_bed(x_opt::Vector{Float64}, 
                                  N::Int,
                                  material_library::Matrix{Float64},
                                  n_segments::Int=3,
                                  material_props::Matrix{Float64}=Matrix{Float64}(undef, 0, 0))

Convert optimization decision variables to MultiMaterialBed.

# Decision Variable Structure (for 3 segments):
- x[1]: Material index 1 (real, 1.0 to n_materials)
- x[2]: Material index 2
- x[3]: Material index 3
- x[4]: Fraction 1 (x1)
- x[5]: Fraction 2 (x2)
- [Remaining variables]: Process variables (times, pressures, etc.)

# Example
```julia
x_opt = [2.3, 5.7, 8.1, 0.3, 0.4, ...]  # mat1, mat2, mat3, frac1, frac2, ...
bed = optimize_decision_vars_to_bed(x_opt[1:5], N, material_library, 3, material_props)
```
"""
function optimize_decision_vars_to_bed(material_vars::Vector{Float64}, 
                                      N::Int,
                                      material_library::Matrix{Float64},
                                      n_segments::Int=3,
                                      material_props::Matrix{Float64}=Matrix{Float64}(undef, 0, 0))
    
    # Extract material indices and fractions
    material_indices = material_vars[1:n_segments]
    fractions_partial = material_vars[n_segments+1:end]
    
    # Compute full fractions (last one is implicit)
    fractions = zeros(n_segments)
    fractions[1:end-1] .= fractions_partial
    fractions[end] = 1.0 - sum(fractions_partial)
    
    return create_multi_material_bed(N, material_indices, fractions, material_library, material_props)
end

export MaterialSegment, MultiMaterialBed
export create_uniform_bed, create_multi_material_bed
export get_isotherm_params, get_segment, get_material_properties
export optimize_decision_vars_to_bed
