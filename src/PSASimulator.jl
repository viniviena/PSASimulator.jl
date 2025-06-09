module PSASimulator

# Include all the module files
include("PSAInput.jl")
include("PSAUtils.jl")
include("StepModels.jl")
include("PSACycle.jl")

# Export the modules
export PSAInput, PSAUtils, StepModels, PSACycleDriver

# Re-export commonly used functions from submodules
using .PSAInput: process_input_parameters
using .PSAUtils: Isotherm
using .StepModels: step_rhs
using .PSACycleDriver: psacycle

export process_input_parameters, Isotherm, step_rhs, psacycle

end # module PSASimulator 