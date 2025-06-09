module StepModels
# =============================================================================
#  StepModels – Interface for all PSA step functions
#  
#  This module provides a unified interface to access all 6 step functions.
#  Modules are loaded once at initialization to avoid repeated loading.
# =============================================================================

export step_rhs

# Get the directory of this file (src/)
const SRC_DIR = dirname(@__FILE__)

# Load all step modules once at initialization
include(joinpath(SRC_DIR, "FuncAdsorption.jl"))
include(joinpath(SRC_DIR, "FuncHeavyReflux.jl"))
include(joinpath(SRC_DIR, "FuncLightReflux.jl"))
include(joinpath(SRC_DIR, "FuncCoCPressurization.jl"))
include(joinpath(SRC_DIR, "FuncCnCDepressurization.jl"))
include(joinpath(SRC_DIR, "FuncCoCDepressurization.jl"))

# Import the modules
using .FuncAdsorptionModule
using .FuncHeavyRefluxModule
using .FuncLightRefluxModule
using .FuncCoCPressurizationModule
using .FuncCnCDepressurizationModule
using .FuncCoCDepressurizationModule

"""
    step_rhs(step_name::Symbol, params, isotherm_params)

Returns the right-hand-side function for the specified PSA step.

Valid step names:
- `:Adsorption`
- `:HeavyReflux`
- `:LightReflux`
- `:CoCPressurization`
- `:CnCDepressurization`
- `:CoCDepressurization`
"""
function step_rhs(step_name::Symbol, params, isotherm_params)
    if step_name == :Adsorption
        return function (du, u, p, t)
            du .= FuncAdsorption(t, u, params, isotherm_params)
        end

    elseif step_name == :HeavyReflux
        return function (du, u, p, t)
            du .= FuncHeavyReflux(t, u, params, isotherm_params)
        end

    elseif step_name == :LightReflux
        return function (du, u, p, t)
            du .= FuncLightReflux(t, u, params, isotherm_params)
        end

    elseif step_name == :CoCPressurization
        return function (du, u, p, t)
            du .= FuncCoCPressurization(t, u, params, isotherm_params)
        end

    elseif step_name == :CnCDepressurization
        return function (du, u, p, t)
            du .= FuncCnCDepressurization(t, u, params, isotherm_params)
        end

    elseif step_name == :CoCDepressurization
        return function (du, u, p, t)
            du .= FuncCoCDepressurization(t, u, params, isotherm_params)
        end

    else
        error("Unknown step: $step_name. Valid steps are: :Adsorption, :HeavyReflux, :LightReflux, :CoCPressurization, :CnCDepressurization, :CoCDepressurization")
    end
end



end # module StepModels 