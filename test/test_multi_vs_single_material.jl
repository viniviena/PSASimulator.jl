using PSASimulator

# Include the source modules
include("../src/FuncAdsorption.jl")
include("../src/FuncAdsorptionMultiMaterial.jl")
include("../src/PSAInput.jl")
include("../src/MaterialSegments.jl")
# Include material property data
include("../demo/demo_data.jl")

using BenchmarkTools
using .FuncAdsorptionModule
using .FuncAdsorptionMultiMaterialModule

conditions = OPT_VARS_PURITY[2, :]

process_vars = [
        1.0,                                        # L [m] - bed length
        conditions[1],                                # P_0 [Pa] - feed pressure  
        conditions[1] * conditions[4] / 8.314 / 313.15, # n_dot_0 [mol/s] - feed molar flow rate
        conditions[2],                                # t_ads [s] - adsorption time
        conditions[3],                                # alpha [-] - light reflux ratio
        conditions[5],                                # beta [-] - heavy reflux ratio
        1.0e4,                                      # P_I [Pa] - intermediate pressure
        conditions[6]                                 # P_l [Pa] - light product pressure
    ]

mat_idx = 2  # Index for material properties (from demodata)
material_iso_params = ISOTHERM_PARAMETERS[mat_idx, :] #From demodata
material_properties = SIMULATION_PARAMETERS[mat_idx, :] #From demodata
material_data = (material_properties, material_iso_params)  #Irrevelant for multibed, but whatever  

# Load standard parameters
params, isotherm_params = process_input_parameters(process_vars, material_data, 40)

N = Int(params[1])

# Create initial state vector (typical initial conditions)
P_init = ones(N + 2)
y_init = 0.15 * ones(N + 2)
x1_init = zeros(N + 2)
x2_init = zeros(N + 2)
T_init = ones(N + 2)

u0 = vcat(P_init, y_init, x1_init, x2_init, T_init)

t = 0.0
du_single = similar(u0)
# Note: FuncAdsorption uses out-of-place signature, so we need to wrap it
derivatives_single = FuncAdsorption(t, u0, params, isotherm_params)

# Test 2: Multi-material with uniform bed
println("Running multi-material function with uniform bed...")

# Create RHS function with cache
rhs! = create_adsorption_rhs_multi_material(params, bed)

# Evaluate
du_multi = similar(u0)

@btime rhs!(du_multi, u0, nothing, t)

# Compare results
println("\n=== Comparing derivatives ===")

# Extract derivative components
dP_single = derivatives_single[1:N+2]
dy_single = derivatives_single[N+3:2*N+4]
dx1_single = derivatives_single[2*N+5:3*N+6]
dx2_single = derivatives_single[3*N+7:4*N+8]
dT_single = derivatives_single[4*N+9:5*N+10]

dP_multi = du_multi[1:N+2]
dy_multi = du_multi[N+3:2*N+4]
dx1_multi = du_multi[2*N+5:3*N+6]
dx2_multi = du_multi[3*N+7:4*N+8]
dT_multi = du_multi[4*N+9:5*N+10]

# Check boundary conditions first
println("Boundary values (should be 0 for most):")
println("  dP[1]: single=$(dP_single[1]), multi=$(dP_multi[1])")
println("  dP[N+2]: single=$(dP_single[N+2]), multi=$(dP_multi[N+2])")
println("  dy[1]: single=$(dy_single[1]), multi=$(dy_multi[1])")
println("  dT[1]: single=$(dT_single[1]), multi=$(dT_multi[1])")

# Check interior points
println("\nFirst interior node (i=2):")
println("  dP[2]: single=$(dP_single[2]), multi=$(dP_multi[2])")
println("  dy[2]: single=$(dy_single[2]), multi=$(dy_multi[2])")
println("  dx1[2]: single=$(dx1_single[2]), multi=$(dx1_multi[2])")
println("  dx2[2]: single=$(dx2_single[2]), multi=$(dx2_multi[2])")
println("  dT[2]: single=$(dT_single[2]), multi=$(dT_multi[2])")

println("\nMiddle node (i=$(div(N,2)+1)):")
mid = div(N,2)+1
println("  dP[$mid]: single=$(dP_single[mid]), multi=$(dP_multi[mid])")
println("  dy[$mid]: single=$(dy_single[mid]), multi=$(dy_multi[mid])")
println("  dx1[$mid]: single=$(dx1_single[mid]), multi=$(dx1_multi[mid])")
println("  dx2[$mid]: single=$(dx2_single[mid]), multi=$(dx2_multi[mid])")
println("  dT[$mid]: single=$(dT_single[mid]), multi=$(dT_multi[mid])")

# Tolerance for floating point comparison
rtol = 1e-10
atol = 1e-12

# Test each component

isapprox(dP_multi, dP_single, rtol=rtol, atol=atol)
max_diff = maximum(abs.(dP_multi .- dP_single))
println("  Max |dP difference|: $max_diff")

@testset "Composition derivatives" begin
    @test isapprox(dy_multi, dy_single, rtol=rtol, atol=atol)
    max_diff = maximum(abs.(dy_multi .- dy_single))
    println("  Max |dy difference|: $max_diff")
end

@testset "Loading derivatives (CO2)" begin
    @test isapprox(dx1_multi, dx1_single, rtol=rtol, atol=atol)
    max_diff = maximum(abs.(dx1_multi .- dx1_single))
    println("  Max |dx1 difference|: $max_diff")
end

@testset "Loading derivatives (N2)" begin
    @test isapprox(dx2_multi, dx2_single, rtol=rtol, atol=atol)
    max_diff = maximum(abs.(dx2_multi .- dx2_single))
    println("  Max |dx2 difference|: $max_diff")
end

@testset "Temperature derivatives" begin
    @test isapprox(dT_multi, dT_single, rtol=rtol, atol=atol)
    max_diff = maximum(abs.(dT_multi .- dT_single))
    println("  Max |dT difference|: $max_diff")
end

# Summary statistics
println("\n=== Summary ===")
println("Total elements: $(length(u0))")
println("Max absolute difference: $(maximum(abs.(du_multi .- derivatives_single)))")
println("Relative error: $(maximum(abs.((du_multi .- derivatives_single) ./ (derivatives_single .+ 1e-16))))")

# Overall test
@test isapprox(du_multi, derivatives_single, rtol=rtol, atol=atol)

println("\nAll tests passed! Multi-material function matches single-material function for uniform bed.")
