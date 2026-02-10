"""
    FuncAdsorptionMultiMaterial.jl

Multi-material version of adsorption step RHS function.
Spatially-varying isotherm parameters based on bed segments.
"""

module FuncAdsorptionMultiMaterialModule

# Get the directory of this file (src/)
const SRC_DIR = dirname(@__FILE__)

# Include required modules
include(joinpath(SRC_DIR, "PSAUtils.jl"))
include(joinpath(SRC_DIR, "MaterialSegments.jl"))

using .PSAUtils
# MaterialSegments.jl is not a module, so we don't need 'using'
# The structs and functions are directly available

# Cache for pre-allocated work arrays (avoids allocation on every RHS call)
struct AdsorptionCache{T <: AbstractVector{<:Real}}
    q_s0_node::T
    deltaU_1_node::T
    deltaU_2_node::T
    q_1_star::T
    q_2_star::T
    sink_term::T
    phi_node::T
    dpdz::T
    dydz::T
    dTdz::T
    d2ydz2::T
    d2Tdz2::T
    vel_cache::VelocityCache{T}
end

function AdsorptionCache(N::Int)
    return AdsorptionCache(
        zeros(N + 2),
        zeros(N + 2),
        zeros(N + 2),
        zeros(N + 2),
        zeros(N + 2),
        zeros(N + 2),
        zeros(N + 2),
        zeros(N),
        zeros(N),
        zeros(N),
        zeros(N),
        zeros(N),
        VelocityCache(N)
    )
end


"""
    FuncAdsorptionMultiMaterial!(du, u, p, t)

In-place RHS evaluation for adsorption step with multi-material bed.
Node-specific isotherm parameters from bed configuration.

# Arguments
- `du`: Pre-allocated derivative vector (modified in-place)
- `u`: State vector [P, y, q1, q2, T] for N+2 volumes
- `p`: Tuple (params, bed, cache) where:
  - params: Process parameters from PSAInput
  - bed: MultiMaterialBed with segment-specific isotherm parameters
  - cache: AdsorptionCache with pre-allocated work arrays
- `t`: Time

# Returns
- `nothing` (modifies `du` in-place)
"""
function FuncAdsorptionMultiMaterial!(du, u, p, t)
    params, bed, cache = p
    
    # Extract process parameters (identical to single-material)
    N = Int(params[1])
    deltaU_1 = params[2]
    deltaU_2 = params[3]
    ro_s = params[4]
    T_0 = params[5]
    epsilon = params[6]
    r_p = params[7]
    mu = params[8]
    R = params[9]
    v_0 = params[10]
    q_s0 = params[11]
    C_pg = params[12]
    C_pa = params[13]
    C_ps = params[14]
    D_m = params[15]
    K_z = params[16]
    P_0 = params[17]
    L = params[18]
    MW_CO2 = params[19]
    MW_N2 = params[20]
    k_1_LDF = params[21]
    k_2_LDF = params[22]
    y_0 = params[23]
    P_inlet = params[26]
    ndot_0 = P_0 / R / T_0 * v_0

    # Extract state variables (slicing creates copies; boundaries will be modified below)
    P = u[1:N+2]
    y = max.(u[N+3:2*N+4], 0.0)
    x1 = max.(u[2*N+5:3*N+6], 0.0)
    x2 = u[3*N+7:4*N+8]
    T = u[4*N+9:5*N+10]

    # Views into pre-allocated derivative vector (in-place, no allocation)
    dPdt  = @view du[1:N+2]
    dydt  = @view du[N+3:2*N+4]
    dx1dt = @view du[2*N+5:3*N+6]
    dx2dt = @view du[3*N+7:4*N+8]
    dTdt  = @view du[4*N+9:5*N+10]

    @views begin # All slicing below becomes views (no copies)

    # Calculate parameters
    dz = 1.0 / N
    D_l = 0.7 * D_m + v_0 * r_p
    Pe = v_0 * L / D_l
    # Note: phi uses params q_s0 (average value) for pressure balance
    # Individual nodes use q_s0_node for isotherm/LDF calculations
    #phi = R * T_0 * q_s0 * (1 - epsilon) / epsilon / P_0
    ro_g = P .* P_0 ./ R ./ T ./ T_0

    # Boundary Conditions
    y[1] = y_0
    T[1] = 1.0  # = 1.0

    # Inlet pressure/velocity handling
    if params[end] == 1
        # Constant pressure inlet
        P[1] = P_inlet
    elseif params[end] == 0
        # Constant velocity inlet
        MW = MW_N2 + (MW_CO2 - MW_N2) * y[1]
        viscous_term = 150.0 * mu * (1.0 - epsilon)^2 / 4.0 / r_p^2 / epsilon^2
        vh_inlet = 1.0
        P[1] = ((viscous_term * vh_inlet * v_0 * dz / 2 * L / P_0) + P[2]) /
               (1.0 - (dz / 2.0 * L / R / T[1] / T_0) * MW * (1.75 * (1.0 - epsilon) / 2.0 / r_p / epsilon) * (vh_inlet * v_0)^2)
    else
        error("Please specify whether inlet velocity or pressure is constant for the feed step")
    end

    # Outlet boundary conditions
    y[N+2] = y[N+1]
    T[N+2] = T[N+1]
    if P[N+1] >= 1
        P[N+2] = 1.0
    else
        P[N+2] = P[N+1]
    end

    # Velocity Calculations (in-place, zero allocations)
    compute_velocity!(cache.vel_cache, P, T, y, params)
    vh = cache.vel_cache.vh
    Ph = cache.vel_cache.Ph
    Th = cache.vel_cache.Th
    yh = cache.vel_cache.yh

    # Spatial derivatives (interior nodes only, length N) - use cache arrays
    dz2_inv = 1.0 / (dz * dz)
    dpdz = cache.dpdz
    dydz = cache.dydz
    dTdz = cache.dTdz
    d2ydz2 = cache.d2ydz2
    d2Tdz2 = cache.d2Tdz2
    
    @. dpdz = (Ph[2:N+1] - Ph[1:N]) / dz
    @. dydz = (yh[2:N+1] - yh[1:N]) / dz
    @. dTdz = (Th[2:N+1] - Th[1:N]) / dz

    d2ydz2[1] = (y[3] - y[2]) * dz2_inv
    @. d2ydz2[2:N-1] = (y[4:N+1] + y[2:N-1] - 2.0 * y[3:N]) * dz2_inv
    d2ydz2[N] = (y[N] - y[N+1]) * dz2_inv

    d2Tdz2[1] = 4 * (Th[2] + T[1] - 2 * T[2]) * dz2_inv
    @. d2Tdz2[2:N-1] = (T[4:N+1] + T[2:N-1] - 2.0 * T[3:N]) * dz2_inv
    d2Tdz2[N] = 4 * (Th[N] + T[N+2] - 2 * T[N+1]) * dz2_inv

    # =====================================================================
    # MULTI-MATERIAL: Node-specific properties, isotherm, LDF & sink term
    # =====================================================================
    
    # Unpack pre-allocated work arrays from cache (no allocation)
    q_s0_node = cache.q_s0_node
    deltaU_1_node = cache.deltaU_1_node
    deltaU_2_node = cache.deltaU_2_node
    q_1_star = cache.q_1_star
    q_2_star = cache.q_2_star
    sink_term = cache.sink_term
    phi_node = cache.phi_node
    
    # LDF rate constants
    k_1 = k_1_LDF * L / v_0
    k_2 = k_2_LDF * L / v_0
    phi_factor = R * T_0 * (1.0 - epsilon) / epsilon / P_0
    
    # Single loop: material properties, isotherm, LDF, sink term, phi
    for i in 1:(N+2)
        seg = bed.is_uniform ? bed.segments[1] : get_segment(bed, i)
        
        q_s0_node[i] = seg.q_s0
        deltaU_1_node[i] = seg.deltaU_1
        deltaU_2_node[i] = seg.deltaU_2
        
        # Equilibrium loadings (Isotherm returns mol/kg; multiply by ro_s → mol/m³)
        q_star = Isotherm(y[i], P[i] * P_0, T[i] * T_0, seg.isotherm_params)
        q_1_star[i] = q_star[1] * seg.ro_s
        q_2_star[i] = q_star[2] * seg.ro_s
        
        # Interior nodes only: LDF, sink term, phi
        if 2 ≤ i ≤ N + 1
            dx1dt[i] = k_1 * (q_1_star[i] / q_s0_node[i] - x1[i])
            dx2dt[i] = k_2 * (q_2_star[i] / q_s0_node[i] - x2[i])
            sink_term[i] = (1.0 - epsilon) * (seg.ro_s * seg.C_ps + q_s0_node[i] * C_pa) +
                           epsilon * ro_g[i] * C_pg
            phi_node[i] = phi_factor * q_s0_node[i]
        end
    end

    # Shared half-cell products (needed across all 3 balances)
    Pv = Ph[1:N+1] .* vh[1:N+1]         # length N+1: 1 alloc
    PvT = Pv ./ Th[1:N+1]               # length N+1: 1 alloc
    ypvt = yh[1:N+1] .* PvT             # length N+1: 1 alloc (needed for dydt)

    # Scalar constants
    transfer_term = K_z / v_0 / L
    adv_coeff = epsilon * C_pg * P_0 / R / T_0
    gen_factor = (1 - epsilon) / T_0

    # 2) Column Energy Balance — single fused write
    @. dTdt[2:N+1] = (
        transfer_term * d2Tdz2 -
        adv_coeff * ((Pv[2:N+1] - Pv[1:N]) - T[2:N+1] * (PvT[2:N+1] - PvT[1:N])) / dz +
        gen_factor * q_s0_node[2:N+1] * (
            (-(deltaU_1_node[2:N+1] - R * T[2:N+1] * T_0)) * dx1dt[2:N+1] +
            (-(deltaU_2_node[2:N+1] - R * T[2:N+1] * T_0)) * dx2dt[2:N+1]
        )
    ) / sink_term[2:N+1]

    # 3) Total mass balance — single fused write
    @. dPdt[2:N+1] = -T[2:N+1] * (PvT[2:N+1] - PvT[1:N]) / dz -
        phi_node[2:N+1] * T[2:N+1] * (dx1dt[2:N+1] + dx2dt[2:N+1]) +
        P[2:N+1] * dTdt[2:N+1] / T[2:N+1]

    # 4) Component Mass Balance — single fused write
    inv_Pe = 1.0 / Pe
    @. dydt[2:N+1] = inv_Pe * (d2ydz2 + dydz * dpdz / P[2:N+1] - dydz * dTdz / T[2:N+1]) -
        (T[2:N+1] / P[2:N+1]) * ((ypvt[2:N+1] - ypvt[1:N]) - y[2:N+1] * (PvT[2:N+1] - PvT[1:N])) / dz +
        (phi_node[2:N+1] * T[2:N+1] / P[2:N+1]) * ((y[2:N+1] - 1) * dx1dt[2:N+1] + y[2:N+1] * dx2dt[2:N+1])

    # Boundary derivatives (must explicitly set all boundaries; du is reused buffer)
    dPdt[1] = 0.0
    dPdt[N+2] = 0.0
    dydt[1] = 0.0
    dydt[N+2] = dydt[N+1]
    dx1dt[1] = 0.0
    dx1dt[N+2] = 0.0
    dx2dt[1] = 0.0
    dx2dt[N+2] = 0.0
    dTdt[1] = 0.0
    dTdt[N+2] = dTdt[N+1]

    end # @views

    return nothing
end

"""
    create_adsorption_rhs_multi_material(params, bed::MultiMaterialBed)

Create a closure for the in-place adsorption RHS with the bed configuration.
Returns a function compatible with DifferentialEquations.jl ODEProblem.
Allocates work arrays once and reuses them across all RHS evaluations.

# Returns
Function with signature: `(du, u, p, t) -> nothing` where p = (params, bed, cache)
"""
function create_adsorption_rhs_multi_material(params, bed)
    N = Int(params[1])
    cache = AdsorptionCache(N)
    p = (params, bed, cache)
    return (du, u, _, t) -> FuncAdsorptionMultiMaterial!(du, u, p, t)
end

export FuncAdsorptionMultiMaterial, create_adsorption_rhs_multi_material, AdsorptionCache

end # module FuncAdsorptionMultiMaterialModule
