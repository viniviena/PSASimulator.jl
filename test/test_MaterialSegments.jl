# ===================================================================
# TEST SUITE FOR MATERIAL SEGMENTS
# ===================================================================
# Tests for multi-material bed configurations using real materials
# from demo_data.jl

using Test
include("../src/MaterialSegments.jl")
include("../demo/demo_data.jl")

# ===================================================================
# HELPER FUNCTIONS
# ===================================================================

"""Extract isotherm parameters (all 13 columns) from ISOTHERM_PARAMETERS"""
function get_isotherm_library()
    # All 13 columns: [q_s_b_1, q_s_d_1, b_1, d_1, dH_b_1, dH_d_1, q_s_b_2, q_s_d_2, b_2, d_2, dH_b_2, dH_d_2, extra]
    return ISOTHERM_PARAMETERS
end

"""Get material properties (ro_s, deltaU_1, deltaU_2) from SIMULATION_PARAMETERS"""
function get_material_properties_library()
    # SIMULATION_PARAMETERS columns: [ro_s, deltaU_1, deltaU_2]
    return SIMULATION_PARAMETERS
end

"""Get material name by index"""
function get_material_name(idx::Int)
    return PURITY_RECOVERY_MATERIALS[idx].name
end

# ===================================================================
# TEST 1: UNIFORM BED (BACKWARD COMPATIBILITY)
# ===================================================================

println("\n" * "="^60)
println("TEST 1: Uniform Bed (Single Material)")
println("="^60)

function test_uniform_bed()
    N = 10
    isotherm_lib = get_isotherm_library()
    material_props = get_material_properties_library()
    
    # Test with Material 1 (Co-MOF-74)
    mat_idx = 1
    isotherm_params = isotherm_lib[mat_idx, :]
    ro_s = material_props[mat_idx, 1]
    
    bed = create_uniform_bed(N, isotherm_params, ro_s)
    
    # Assertions
    @test bed.N == N
    @test bed.is_uniform == true
    @test length(bed.segments) == 1
    @test bed.segments[1].start_node == 1
    @test bed.segments[1].end_node == N
    @test bed.segments[1].fraction ≈ 1.0
    @test bed.segments[1].isotherm_params == isotherm_params
    @test bed.segments[1].ro_s == ro_s
    @test bed.segments[1].C_ps == 1070.0  # Default value
    @test bed.segments[1].q_s0 ≈ 5.84 * ro_s  # q_s0 = q_s × ρ_s
    
    # Test parameter retrieval for all nodes
    for node in 1:N
        params = get_isotherm_params(bed, node)
        @test params == isotherm_params
        
        ro_s_node, C_ps_node, q_s0_node = get_material_properties(bed, node)
        @test ro_s_node == ro_s
        @test C_ps_node == 1070.0
        @test q_s0_node ≈ 5.84 * ro_s
    end
    
    println("✓ Material: $(get_material_name(mat_idx))")
    println("✓ Nodes: 1 to $N")
    println("✓ Isotherm params: $(round.(isotherm_params, sigdigits=4))")
    println("✓ Solid density: $ro_s kg/m³")
    println("✓ Solid heat capacity: 1070.0 J/(kg·K)")
    println("✓ Volumetric saturation: $(round(5.84 * ro_s, digits=1)) mol/m³")
    println("✓ All nodes return same parameters")
    println("✓ is_uniform flag: $(bed.is_uniform)")
    
    return bed
end

bed_uniform = test_uniform_bed()

# ===================================================================
# TEST 2: THREE-SEGMENT BED WITH INTEGER MATERIAL INDICES
# ===================================================================

println("\n" * "="^60)
println("TEST 2: Three-Segment Bed (Integer Material Indices)")
println("="^60)

function test_three_segment_integer()
    N = 30
    isotherm_lib = get_isotherm_library()
    material_props = get_material_properties_library()
    
    # Materials: Mg-MOF-74, Ni-MOF-74, Zeolite 13X
    material_indices = [4.0, 6.0, 16.0]
    fractions = [0.3, 0.4, 0.3]
    
    bed = create_multi_material_bed(N, material_indices, fractions, isotherm_lib, material_props)
    
    # Expected node ranges
    # Segment 1: nodes 1-9   (30 * 0.3 = 9)
    # Segment 2: nodes 10-21 (30 * 0.7 = 21, so 10 to 21)
    # Segment 3: nodes 22-30 (remaining)
    
    @test bed.N == N
    @test bed.is_uniform == false
    @test length(bed.segments) == 3
    
    # Check segment 1
    seg1 = bed.segments[1]
    @test seg1.start_node == 1
    @test seg1.end_node == 9
    @test seg1.fraction ≈ 0.3
    @test round(Int, seg1.material_idx) == 4
    
    # Check segment 2
    seg2 = bed.segments[2]
    @test seg2.start_node == 10
    @test seg2.end_node == 21
    @test seg2.fraction ≈ 0.4
    @test round(Int, seg2.material_idx) == 6
    
    # Check segment 3
    seg3 = bed.segments[3]
    @test seg3.start_node == 22
    @test seg3.end_node == 30
    @test seg3.fraction ≈ 0.3
    @test round(Int, seg3.material_idx) == 16
    
    println("✓ Total nodes: $N")
    println("\nSegment 1: Nodes $(seg1.start_node)-$(seg1.end_node)")
    println("  Material: $(get_material_name(4)) (index $(seg1.material_idx))")
    println("  Fraction: $(seg1.fraction)")
    println("  Params: $(round.(seg1.isotherm_params, sigdigits=4))")
    
    println("\nSegment 2: Nodes $(seg2.start_node)-$(seg2.end_node)")
    println("  Material: $(get_material_name(6)) (index $(seg2.material_idx))")
    println("  Fraction: $(seg2.fraction)")
    println("  Params: $(round.(seg2.isotherm_params, sigdigits=4))")
    
    println("\nSegment 3: Nodes $(seg3.start_node)-$(seg3.end_node)")
    println("  Material: $(get_material_name(16)) (index $(seg3.material_idx))")
    println("  Fraction: $(seg3.fraction)")
    println("  Params: $(round.(seg3.isotherm_params, sigdigits=4))")
    
    # Test parameter retrieval at specific nodes
    params_node5 = get_isotherm_params(bed, 5)
    params_node15 = get_isotherm_params(bed, 15)
    params_node25 = get_isotherm_params(bed, 25)
    
    @test params_node5 == seg1.isotherm_params
    @test params_node15 == seg2.isotherm_params
    @test params_node25 == seg3.isotherm_params
    
    println("\n✓ Parameter lookup verified:")
    println("  Node 5  -> Material $(get_material_name(4))")
    println("  Node 15 -> Material $(get_material_name(6))")
    println("  Node 25 -> Material $(get_material_name(16))")
    
    return bed
end

bed_three_int = test_three_segment_integer()

# ===================================================================
# TEST 3: THREE-SEGMENT BED WITH FRACTIONAL MATERIAL INDICES
# ===================================================================

println("\n" * "="^60)
println("TEST 3: Three-Segment Bed (Fractional Material Indices)")
println("="^60)

function test_three_segment_fractional()
    N = 30
    isotherm_lib = get_isotherm_library()
    material_props = get_material_properties_library()
    
    # Fractional indices: 2.3 -> Material 2, 5.7 -> Material 6, 9.1 -> Material 9
    material_indices = [2.3, 5.7, 9.1]
    fractions = [0.25, 0.50, 0.25]
    
    bed = create_multi_material_bed(N, material_indices, fractions, isotherm_lib, material_props)
    
    @test bed.N == N
    @test bed.is_uniform == false
    @test length(bed.segments) == 3
    
    # Check rounding behavior
    seg1 = bed.segments[1]
    seg2 = bed.segments[2]
    seg3 = bed.segments[3]
    
    @test round(Int, seg1.material_idx) == 2  # 2.3 rounds to 2
    @test round(Int, seg2.material_idx) == 6  # 5.7 rounds to 6
    @test round(Int, seg3.material_idx) == 9  # 9.1 rounds to 9
    
    println("✓ Fractional index rounding:")
    println("  $(material_indices[1]) -> Material $(round(Int, seg1.material_idx)) ($(get_material_name(2)))")
    println("  $(material_indices[2]) -> Material $(round(Int, seg2.material_idx)) ($(get_material_name(6)))")
    println("  $(material_indices[3]) -> Material $(round(Int, seg3.material_idx)) ($(get_material_name(9)))")
    
    println("\n✓ Segment ranges:")
    println("  Segment 1: Nodes $(seg1.start_node)-$(seg1.end_node) ($(fractions[1]*100)%)")
    println("  Segment 2: Nodes $(seg2.start_node)-$(seg2.end_node) ($(fractions[2]*100)%)")
    println("  Segment 3: Nodes $(seg3.start_node)-$(seg3.end_node) ($(fractions[3]*100)%)")
    
    return bed
end

bed_three_frac = test_three_segment_fractional()

# ===================================================================
# TEST 4: EDGE CASES - BOUNDARY VALUES
# ===================================================================

println("\n" * "="^60)
println("TEST 4: Edge Cases and Boundary Conditions")
println("="^60)

function test_edge_cases()
    N = 10
    isotherm_lib = get_isotherm_library()
    material_props = get_material_properties_library()
    
    println("\n📋 Test 4a: Material indices at boundaries (1.0, 9.5, 16.0)")
    # Test boundary material indices
    material_indices = [1.0, 9.5, 16.0]  # First, middle rounded up, last
    fractions = [0.33, 0.33, 0.34]
    
    bed = create_multi_material_bed(N, material_indices, fractions, isotherm_lib, material_props)
    
    @test round(Int, bed.segments[1].material_idx) == 1
    @test round(Int, bed.segments[2].material_idx) == 10  # 9.5 rounds to 10
    @test round(Int, bed.segments[3].material_idx) == 16
    
    println("  ✓ Index 1.0 -> Material 1 ($(get_material_name(1)))")
    println("  ✓ Index 9.5 -> Material 10 ($(get_material_name(10)))")
    println("  ✓ Index 16.0 -> Material 16 ($(get_material_name(16)))")
    
    println("\n📋 Test 4b: Unequal fractions [0.1, 0.6, 0.3]")
    material_indices = [3.0, 12.0, 15.0]
    fractions = [0.1, 0.6, 0.3]
    
    bed2 = create_multi_material_bed(N, material_indices, fractions, isotherm_lib, material_props)
    
    seg1 = bed2.segments[1]
    seg2 = bed2.segments[2]
    seg3 = bed2.segments[3]
    
    println("  ✓ Segment 1: $(seg1.end_node - seg1.start_node + 1) nodes ($(fractions[1]*100)%)")
    println("  ✓ Segment 2: $(seg2.end_node - seg2.start_node + 1) nodes ($(fractions[2]*100)%)")
    println("  ✓ Segment 3: $(seg3.end_node - seg3.start_node + 1) nodes ($(fractions[3]*100)%)")
    
    # Verify all nodes are covered
    @test seg1.start_node == 1
    @test seg3.end_node == N
    @test seg2.start_node == seg1.end_node + 1
    @test seg3.start_node == seg2.end_node + 1
    
    println("\n📋 Test 4c: Very small fraction [0.05, 0.05, 0.9]")
    fractions_small = [0.05, 0.05, 0.9]
    bed3 = create_multi_material_bed(20, material_indices, fractions_small, isotherm_lib, material_props)
    
    # Even with small fractions, should have valid node assignments
    @test bed3.segments[1].end_node >= bed3.segments[1].start_node
    @test bed3.segments[2].end_node >= bed3.segments[2].start_node
    @test bed3.segments[3].end_node >= bed3.segments[3].start_node
    @test bed3.segments[3].end_node == 20
    
    println("  ✓ Small fractions handled correctly")
    println("    Seg 1: $(bed3.segments[1].end_node - bed3.segments[1].start_node + 1) nodes")
    println("    Seg 2: $(bed3.segments[2].end_node - bed3.segments[2].start_node + 1) nodes")
    println("    Seg 3: $(bed3.segments[3].end_node - bed3.segments[3].start_node + 1) nodes")
    
    return bed, bed2, bed3
end

beds_edge = test_edge_cases()

# ===================================================================
# TEST 5: OPTIMIZATION VARIABLE DECODING
# ===================================================================

println("\n" * "="^60)
println("TEST 5: Optimization Variable Decoding")
println("="^60)

function test_optimize_decision_vars()
    N = 30
    isotherm_lib = get_isotherm_library()
    material_props = get_material_properties_library()
    n_segments = 3
    
    # Simulate optimizer output for 3 segments
    # [mat1, mat2, mat3, frac1, frac2]
    material_vars = [4.2, 6.8, 15.3, 0.35, 0.30]
    # frac3 = 1 - 0.35 - 0.30 = 0.35 (implicit)
    
    bed = optimize_decision_vars_to_bed(material_vars, N, isotherm_lib, n_segments, material_props)
    
    @test bed.N == N
    @test length(bed.segments) == 3
    
    # Check material rounding
    @test round(Int, bed.segments[1].material_idx) == 4   # 4.2 -> 4
    @test round(Int, bed.segments[2].material_idx) == 7   # 6.8 -> 7
    @test round(Int, bed.segments[3].material_idx) == 15  # 15.3 -> 15
    
    # Check fractions
    @test bed.segments[1].fraction ≈ 0.35
    @test bed.segments[2].fraction ≈ 0.30
    @test bed.segments[3].fraction ≈ 0.35  # Implicit
    
    println("✓ Decision variable decoding:")
    println("  Input: mat_idx = [4.2, 6.8, 15.3], frac = [0.35, 0.30, implicit]")
    println("\n  Decoded bed configuration:")
    for (i, seg) in enumerate(bed.segments)
        mat_idx = round(Int, seg.material_idx)
        println("    Segment $i: $(get_material_name(mat_idx)) " *
                "(nodes $(seg.start_node)-$(seg.end_node), " *
                "fraction=$(round(seg.fraction, digits=2)))")
    end
    
    return bed
end

bed_decoded = test_optimize_decision_vars()

# ===================================================================
# TEST 6: REALISTIC OPTIMIZATION SCENARIO
# ===================================================================

println("\n" * "="^60)
println("TEST 6: Realistic Optimization Scenario")
println("="^60)

function test_realistic_scenario()
    N = 50  # More realistic discretization
    isotherm_lib = get_isotherm_library()
    material_props = get_material_properties_library()
    
    println("Scenario: Optimizer explores different material combinations\n")
    
    # Scenario 1: High selectivity material at inlet
    println("📊 Scenario 1: High-selectivity inlet, moderate middle, adsorbent outlet")
    material_indices_1 = [10.0, 6.0, 16.0]  # SIFSIX-3-Ni, Ni-MOF-74, Zeolite 13X
    fractions_1 = [0.20, 0.40, 0.40]
    
    bed1 = create_multi_material_bed(N, material_indices_1, fractions_1, isotherm_lib, material_props)
    
    println("  Configuration:")
    for (i, seg) in enumerate(bed1.segments)
        mat_idx = round(Int, seg.material_idx)
        n_nodes = seg.end_node - seg.start_node + 1
        println("    Layer $i: $(get_material_name(mat_idx))")
        println("      Nodes: $(seg.start_node)-$(seg.end_node) ($n_nodes nodes, $(round(seg.fraction*100, digits=1))%)")
        println("      b_inf: [$(round(seg.isotherm_params[1], sigdigits=3)), " *
                "$(round(seg.isotherm_params[2], sigdigits=3))]")
    end
    
    # Scenario 2: Gradual selectivity gradient
    println("\n📊 Scenario 2: Gradual selectivity gradient")
    material_indices_2 = [4.5, 7.2, 14.9]  # MOF materials with different properties
    fractions_2 = [0.33, 0.34, 0.33]
    
    bed2 = create_multi_material_bed(N, material_indices_2, fractions_2, isotherm_lib, material_props)
    
    println("  Configuration:")
    for (i, seg) in enumerate(bed2.segments)
        mat_idx = round(Int, seg.material_idx)
        n_nodes = seg.end_node - seg.start_node + 1
        println("    Layer $i: $(get_material_name(mat_idx))")
        println("      Nodes: $(seg.start_node)-$(seg.end_node) ($n_nodes nodes)")
        println("      dH: [$(round(seg.isotherm_params[3], digits=0)), " *
                "$(round(seg.isotherm_params[4], digits=0))] J/mol")
    end
    
    return bed1, bed2
end

beds_realistic = test_realistic_scenario()

# ===================================================================
# TEST 7: PERFORMANCE CHECK
# ===================================================================

println("\n" * "="^60)
println("TEST 7: Performance Verification")
println("="^60)

function test_performance()
    N = 100
    isotherm_lib = get_isotherm_library()
    material_props = get_material_properties_library()
    
    # Test uniform bed (fast path)
    println("⏱️  Uniform bed parameter lookup (1,000 calls):")
    bed_uniform = create_uniform_bed(N, isotherm_lib[1, :])
    
    time_uniform = @elapsed begin
        for _ in 1:1000
            for node in 1:N
                params = get_isotherm_params(bed_uniform, node)
            end
        end
    end
    
    println("  Time: $(round(time_uniform*1000, digits=2)) ms")
    println("  is_uniform flag: $(bed_uniform.is_uniform)")
    
    # Test multi-material bed
    println("\n⏱️  Multi-material bed parameter lookup (1,000 calls):")
    bed_multi = create_multi_material_bed(N, [2.0, 6.0, 12.0], [0.3, 0.4, 0.3], isotherm_lib, material_props)
    
    time_multi = @elapsed begin
        for _ in 1:1000
            for node in 1:N
                params = get_isotherm_params(bed_multi, node)
            end
        end
    end
    
    println("  Time: $(round(time_multi*1000, digits=2)) ms")
    println("  is_uniform flag: $(bed_multi.is_uniform)")
    println("  Overhead: $(round((time_multi/time_uniform - 1)*100, digits=1))%")
    
    return time_uniform, time_multi
end

times = test_performance()

# ===================================================================
# SUMMARY
# ===================================================================

println("\n" * "="^60)
println("✅ ALL TESTS PASSED")
println("="^60)
println("\nTest Summary:")
println("  ✓ Test 1: Uniform bed (backward compatibility)")
println("  ✓ Test 2: Three-segment bed with integer indices")
println("  ✓ Test 3: Three-segment bed with fractional indices")
println("  ✓ Test 4: Edge cases and boundary conditions")
println("  ✓ Test 5: Optimization variable decoding")
println("  ✓ Test 6: Realistic optimization scenarios")
println("  ✓ Test 7: Performance verification")
println("\n" * "="^60)
