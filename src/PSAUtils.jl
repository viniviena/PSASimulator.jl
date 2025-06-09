module PSAUtils
# =========================================================================
#  Weighted‑Essentially‑Non‑Oscillatory (WENO) wall‑flux evaluator
#  -------------------------------------------------------------------------
#  Port of WENO.m (Karson thesis PSA simulator) from MATLAB → Julia.
#  Behaviour, indexing, and edge handling are kept **identical** to the
#  original so results remain bit‑for‑bit comparable.
# =========================================================================
export weno, Isotherm, isotherm_equilibrium, compute_velocity, trapz, WENO          # main entry point (vector or matrix)

# ──────────────────────────────────────────────────────────────────────────
#  Helper: promote 1‑D inputs to 2‑D, run core, then unwrap.
# ──────────────────────────────────────────────────────────────────────────
function weno(flux_c::AbstractVector{T}, flowdir::AbstractString) where {T}
    fw = weno(reshape(flux_c, :, 1), flowdir)
    return vec(fw)                   # drop the singleton second dimension
end

# ──────────────────────────────────────────────────────────────────────────
#  Core 2‑D implementation (N + 2 rows × m columns, like MATLAB code)
# ──────────────────────────────────────────────────────────────────────────
function weno(flux_c::AbstractMatrix{T}, flowdir::AbstractString) where {T}
    oo = T(1e-10)   # small number to avoid /0 - fixed type conversion
    N, m = size(flux_c)
    N -= 2                               # internal cell count
    flux_w = zeros(T, N + 1, m)              # output (walls)
    alpha0 = similar(flux_c)
    alpha1 = similar(flux_c)

    # Boundary walls – identical to MATLAB
    flux_w[1, :] .= flux_c[1, :]
    flux_w[N+1, :] .= flux_c[N+2, :]

    # Choose scheme
    dir = lowercase(strip(flowdir))
    if dir == "upwind"
        @views begin
            # Smoothness coefficients
            alpha0[2:N, :] .= (2 / 3) ./ ((flux_c[3:N+1, :] .- flux_c[2:N, :] .+ oo) .^ 4)
            alpha1[3:N, :] .= (1 / 3) ./ ((flux_c[3:N, :] .- flux_c[2:N-1, :] .+ oo) .^ 4)
            alpha1[2, :] .= (1 / 3) ./ ((2 .* (flux_c[2, :] .- flux_c[1, :]) .+ oo) .^ 4)

            # Interior walls j = 3 … N‑1
            flux_w[3:N, :] .= (alpha0[3:N, :] ./ (alpha0[3:N, :] .+ alpha1[3:N, :])) .* 0.5 .* (flux_c[3:N, :] .+ flux_c[4:N+1, :]) .+
                              (alpha1[3:N, :] ./ (alpha0[3:N, :] .+ alpha1[3:N, :])) .* (1.5 .* flux_c[3:N, :] .- 0.5 .* flux_c[2:N-1, :])

            # Wall j = 2
            flux_w[2, :] .= (alpha0[2, :] ./ (alpha0[2, :] .+ alpha1[2, :])) .* 0.5 .* (flux_c[2, :] .+ flux_c[3, :]) .+
                            (alpha1[2, :] ./ (alpha0[2, :] .+ alpha1[2, :])) .* (2 .* flux_c[2, :] .- flux_c[1, :])
        end

    elseif dir == "downwind"
        @views begin
            alpha0[2:N, :] .= (2 / 3) ./ ((flux_c[2:N, :] .- flux_c[3:N+1, :] .+ oo) .^ 4)
            alpha1[2:N-1, :] .= (1 / 3) ./ ((flux_c[3:N, :] .- flux_c[4:N+1, :] .+ oo) .^ 4)
            alpha1[N, :] .= (1 / 3) ./ ((2 .* (flux_c[N+1, :] .- flux_c[N+2, :]) .+ oo) .^ 4)

            flux_w[2:N-1, :] .= (alpha0[2:N-1, :] ./ (alpha0[2:N-1, :] .+ alpha1[2:N-1, :])) .* 0.5 .* (flux_c[2:N-1, :] .+ flux_c[3:N, :]) .+
                                (alpha1[2:N-1, :] ./ (alpha0[2:N-1, :] .+ alpha1[2:N-1, :])) .* (1.5 .* flux_c[3:N, :] .- 0.5 .* flux_c[4:N+1, :])

            flux_w[N, :] .= (alpha0[N, :] ./ (alpha0[N, :] .+ alpha1[N, :])) .* 0.5 .* (flux_c[N, :] .+ flux_c[N+1, :]) .+
                            (alpha1[N, :] ./ (alpha0[N, :] .+ alpha1[N, :])) .* (2 .* flux_c[N+1, :] .- flux_c[N+2, :])
        end
    else
        error("flowdir must be \"upwind\" or \"downwind\" (got \"$flowdir\")")
    end

    return flux_w
end

# Add WENO alias for consistency with existing code
const WENO = weno

# ──────────────────────────────────────────────────────────────────────────
#  Isotherm calculation - Direct port from MATLAB Isotherm.m
# ──────────────────────────────────────────────────────────────────────────
function Isotherm(y::AbstractVector, P::AbstractVector, T::AbstractVector, isotherm_parameters::AbstractVector)
    """
    Calculate the molar loadings of a two component mixture using dual site
    Langmuir competitive isotherm with temperature dependent parameters.

    Args:
        y: mole fraction of component one [-]. Component 2 mole fraction is 1-y
        P: Total pressure of the gas [Pa]
        T: Temperature of the gas [K]
        isotherm_parameters: Parameters for the isotherm (length 13)

    Returns:
        q: Matrix with columns [q1, q2] containing molar loadings [mol/kg]
    """
    R = 8.314  # J/(mol·K)

    # Extract parameters
    q_s_b_1 = isotherm_parameters[1]
    q_s_d_1 = isotherm_parameters[3]
    q_s_b_2 = isotherm_parameters[2]
    q_s_d_2 = isotherm_parameters[4]
    b_1 = isotherm_parameters[5]
    d_1 = isotherm_parameters[7]
    b_2 = isotherm_parameters[6]
    d_2 = isotherm_parameters[8]
    deltaU_b_1 = isotherm_parameters[9]
    deltaU_d_1 = isotherm_parameters[11]
    deltaU_b_2 = isotherm_parameters[10]
    deltaU_d_2 = isotherm_parameters[12]

    # Temperature-dependent parameters
    B_1 = b_1 .* exp.(-deltaU_b_1 ./ R ./ T)
    D_1 = d_1 .* exp.(-deltaU_d_1 ./ R ./ T)
    B_2 = b_2 .* exp.(-deltaU_b_2 ./ R ./ T)
    D_2 = d_2 .* exp.(-deltaU_d_2 ./ R ./ T)

    # Choose input type based on parameter 13
    if isotherm_parameters[13] == 0
        # Partial pressure
        P_1 = y .* P
        P_2 = (1 .- y) .* P
        input_1 = P_1
        input_2 = P_2
    elseif isotherm_parameters[13] == 1
        # Concentration
        C_1 = y .* P ./ R ./ T
        C_2 = (1 .- y) .* P ./ R ./ T
        input_1 = C_1
        input_2 = C_2
    else
        error("Please specify whether the isotherms are in terms of Concentration or Partial Pressure")
    end

    # Calculate loadings for both sites
    q1_b = q_s_b_1 .* B_1 .* input_1 ./ (1 .+ B_1 .* input_1 .+ B_2 .* input_2)
    q1_d = q_s_d_1 .* D_1 .* input_1 ./ (1 .+ D_1 .* input_1 .+ D_2 .* input_2)
    q1 = q1_b .+ q1_d

    q2_b = q_s_b_2 .* B_2 .* input_2 ./ (1 .+ B_1 .* input_1 .+ B_2 .* input_2)
    q2_d = q_s_d_2 .* D_2 .* input_2 ./ (1 .+ D_1 .* input_1 .+ D_2 .* input_2)
    q2 = q2_b .+ q2_d

    return hcat(q1, q2)
end

# Convenience function for scalar inputs
function Isotherm(y::Real, P::Real, T::Real, isotherm_parameters::AbstractVector)
    return Isotherm([y], [P], [T], isotherm_parameters)
end

# Alias for the function name used in the step functions
function isotherm_equilibrium(y::Real, P::Real, T::Real, isotherm_parameters::AbstractVector)
    q = Isotherm(y, P, T, isotherm_parameters)
    return q[1, 1], q[1, 2]  # Return as tuple (q1, q2)
end

# ──────────────────────────────────────────────────────────────────────────
#  Ergun equation velocity calculator - Enhanced for all PSA steps
# ──────────────────────────────────────────────────────────────────────────
function compute_velocity(P::AbstractVector, T::AbstractVector, y::AbstractVector, Params::AbstractVector;
    weno_scheme::String="upwind", adaptive_weno::Bool=false,
    Ph_in::Union{AbstractVector,Nothing}=nothing,
    Th_in::Union{AbstractVector,Nothing}=nothing,
    yh_in::Union{AbstractVector,Nothing}=nothing)
    """
    Calculate interstitial velocity using Ergun equation.
    Enhanced version that handles all PSA step requirements.

    Args:
        P: Dimensionless pressure [N+2]
        T: Dimensionless temperature [N+2]  
        y: Gas mole fraction [N+2]
        Params: Parameter vector from PSAInput
        weno_scheme: "upwind" or "downwind" for WENO scheme (default: "upwind")
        adaptive_weno: Whether to use adaptive WENO based on pressure changes (default: false)
        Ph_in, Th_in, yh_in: Pre-computed wall values (optional, for adaptive schemes)

    Returns:
        vh: Dimensionless velocity at cell walls [N+1]
        Ph, Th, yh: Values at walls (for reuse in calling function)
        dpdzh: Pressure gradient at walls
    """
    # Extract parameters
    N = Int(Params[1])
    epsilon = Params[6]
    r_p = Params[7]
    mu = Params[8]
    P_0 = Params[17]
    L = Params[18]
    MW_CO2 = Params[19]
    MW_N2 = Params[20]
    v_0 = Params[10]
    R = Params[9]
    T_0 = Params[5]

    dz = 1.0 / N

    # Get values at walls using WENO or provided values
    if Ph_in !== nothing && Th_in !== nothing && yh_in !== nothing
        # Use provided wall values (for adaptive schemes)
        Ph = Ph_in
        Th = Th_in
        yh = yh_in
    elseif adaptive_weno
        # Adaptive WENO based on pressure changes
        dP_vec = P[2:N+2] - P[1:N+1]
        idx_f = findall(dP_vec .<= 0.0)  # Forward flow
        idx_b = findall(dP_vec .> 0.0)   # Backward flow

        # Pressure at walls
        Ph = zeros(N + 1)
        Ph_f = weno(P, "upwind")
        Ph_b = weno(P, "downwind")
        Ph[idx_f] = Ph_f[idx_f]
        Ph[idx_b] = Ph_b[idx_b]
        Ph[1] = P[1]
        Ph[N+1] = P[N+2]

        # Mole fraction at walls
        yh = zeros(N + 1)
        yh_f = weno(y, "upwind")
        yh_b = weno(y, "downwind")
        yh[idx_f] = yh_f[idx_f]
        yh[idx_b] = yh_b[idx_b]
        # Special handling for boundaries
        if P[1] > P[2]
            yh[1] = y[1]
        else
            yh[1] = y[2]
        end
        yh[N+1] = y[N+2]

        # Temperature at walls
        Th = zeros(N + 1)
        Th_f = weno(T, "upwind")
        Th_b = weno(T, "downwind")
        Th[idx_f] = Th_f[idx_f]
        Th[idx_b] = Th_b[idx_b]
        # Special handling for boundaries
        if P[1] > P[2]
            Th[1] = T[1]
        else
            Th[1] = T[2]
        end
        Th[N+1] = T[N+2]
    else
        # Standard WENO (upwind or downwind)
        Ph = weno(P, weno_scheme)
        Th = weno(T, weno_scheme)
        yh = weno(y, weno_scheme)
    end

    # Calculate pressure gradient at walls
    dpdzh = zeros(N + 1)
    dpdzh[2:N] .= (P[3:N+1] .- P[2:N]) ./ dz
    dpdzh[1] = 2 * (P[2] - P[1]) / dz
    dpdzh[N+1] = 2 * (P[N+2] - P[N+1]) / dz

    # Gas density at walls
    ro_gh = (P_0 / R / T_0) .* Ph[1:N+1] ./ Th[1:N+1]

    # Ergun equation coefficients
    viscous_term = 150 * mu * (1 - epsilon)^2 / 4 / r_p^2 / epsilon^2

    # Molecular weight at walls
    MW_h = MW_N2 .+ (MW_CO2 - MW_N2) .* yh[1:N+1]
    kinetic_term_h = ro_gh .* MW_h .* (1.75 * (1 - epsilon) / 2 / r_p / epsilon)

    # Solve Ergun equation for velocity
    vh = zeros(N + 1)
    for i in 1:N+1
        if abs(kinetic_term_h[i]) > 1e-10  # Avoid division by zero
            discriminant = viscous_term^2 + 4 * kinetic_term_h[i] * abs(dpdzh[i]) * P_0 / L
            # Add safeguard for numerical precision issues
            discriminant = max(discriminant, 0.0)
            vh[i] = -sign(dpdzh[i]) * (-viscous_term + sqrt(discriminant)) / (2 * kinetic_term_h[i]) / v_0
        else
            # Fallback for very low density
            vh[i] = 0.0
        end
    end

    return vh, Ph, Th, yh, dpdzh
end

# Simplified interface for basic velocity calculation - renamed to avoid method conflict
function compute_velocity_simple(P::AbstractVector, T::AbstractVector, y::AbstractVector, Params::AbstractVector)
    """
    Simple interface for velocity calculation using upwind WENO.
    Returns only velocity, not the wall values.
    """
    vh, _, _, _, _ = compute_velocity(P, T, y, Params; weno_scheme="upwind")
    return vh
end

# ──────────────────────────────────────────────────────────────────────────
#  Trapezoidal integration - matches MATLAB trapz function
# ──────────────────────────────────────────────────────────────────────────
function trapz(t::AbstractVector, y::AbstractVector)
    """
    Trapezoidal numerical integration, identical to MATLAB trapz.
    """
    @assert length(t) == length(y) "t and y must have same length"
    if length(t) < 2
        return 0.0
    end
    return sum(diff(t) .* (y[1:end-1] .+ y[2:end]) ./ 2)
end

function trapz(y::AbstractVector)
    """
    Trapezoidal integration with unit spacing.
    """
    if length(y) < 2
        return 0.0
    end
    return sum((y[1:end-1] .+ y[2:end]) ./ 2)
end

end # module PSAUtils
