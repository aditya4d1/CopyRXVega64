    .text
    .hsa_code_object_version 2,1
    .hsa_code_object_isa 9,0,0,"AMD","AMDGPU"
    .weak   foo_64_2_32_1024 ; -- Begin function foo_64_2_32_1024
    .p2align    8
    .type   foo_64_2_32_1024,@function
    .amdgpu_hsa_kernel foo_64_2_32_1024
foo_64_2_32_1024:           ; @foo_64_2_32_1024
.Lfunc_begin0:
    .amd_kernel_code_t
        amd_code_version_major = 1
        amd_code_version_minor = 1
        amd_machine_kind = 1
        amd_machine_version_major = 9
        amd_machine_version_minor = 0
        amd_machine_version_stepping = 0
        kernel_code_entry_byte_offset = 256
        kernel_code_prefetch_byte_size = 0
        max_scratch_backing_memory_byte_size = 0
        granulated_workitem_vgpr_count = 15
        granulated_wavefront_sgpr_count = 1
        priority = 0
        float_mode = 192
        priv = 0
        enable_dx10_clamp = 1
        debug_mode = 0
        enable_ieee_mode = 1
        enable_sgpr_private_segment_wave_byte_offset = 0
        user_sgpr_count = 6
        enable_trap_handler = 1
        enable_sgpr_workgroup_id_x = 1
        enable_sgpr_workgroup_id_y = 0
        enable_sgpr_workgroup_id_z = 0
        enable_sgpr_workgroup_info = 0
        enable_vgpr_workitem_id = 0
        enable_exception_msb = 0
        granulated_lds_size = 0
        enable_exception = 0
        enable_sgpr_private_segment_buffer = 1
        enable_sgpr_dispatch_ptr = 0
        enable_sgpr_queue_ptr = 0
        enable_sgpr_kernarg_segment_ptr = 1
        enable_sgpr_dispatch_id = 0
        enable_sgpr_flat_scratch_init = 0
        enable_sgpr_private_segment_size = 0
        enable_sgpr_grid_workgroup_count_x = 0
        enable_sgpr_grid_workgroup_count_y = 0
        enable_sgpr_grid_workgroup_count_z = 0
        enable_ordered_append_gds = 0
        private_element_size = 1
        is_ptr64 = 1
        is_dynamic_callstack = 0
        is_debug_enabled = 0
        is_xnack_enabled = 0
        workitem_private_segment_byte_size = 0
        workgroup_group_segment_byte_size = 0
        gds_segment_byte_size = 0
        kernarg_segment_byte_size = 16
        workgroup_fbarrier_count = 0
        wavefront_sgpr_count = 16
        workitem_vgpr_count = 64
        reserved_vgpr_first = 0
        reserved_vgpr_count = 0
        reserved_sgpr_first = 0
        reserved_sgpr_count = 0
        debug_wavefront_private_segment_offset_sgpr = 0
        debug_private_segment_buffer_sgpr = 0
        kernarg_segment_alignment = 4
        group_segment_alignment = 4
        private_segment_alignment = 4
        wavefront_size = 6
        call_convention = -1
        runtime_loader_kernel_symbol = 0
    .end_amd_kernel_code_t
; %bb.0:                                ; %entry

.set sgpr_args, 4
.set sgpr_input_ptr, 0
.set sgpr_output_ptr, 2
.set sgpr_stride_bytes, 4
.set sgpr_num_loops, 5
.set sgpr_loop_idx, 6

.set vgpr_thread_idx, 0
.set vgpr_float4_idx, 1
.set vgpr_input_ptr, 2
.set vgpr_output_ptr, 4
.set vgpr_data, 6

.set k_num_workitems, 1024
.set k_num_loops, 32

//! We use unroll factor = 2

    s_load_dwordx2 s[sgpr_input_ptr:sgpr_input_ptr+1], s[sgpr_args:sgpr_args+1], 0x0
    s_load_dwordx2 s[sgpr_output_ptr:sgpr_output_ptr+1], s[sgpr_args:sgpr_args+1], 0x8

    s_mov_b32 s[sgpr_stride_bytes], k_num_workitems * 16 // bytes
    s_mov_b32 s[sgpr_num_loops], k_num_loops
    s_mov_b32 s[sgpr_loop_idx], 2
    v_lshlrev_b32 v[vgpr_float4_idx], 4, v[vgpr_thread_idx]

    //! Wait for input pointer
    s_waitcnt lgkmcnt(1)

    v_mov_b32 v[vgpr_input_ptr+1], s[sgpr_input_ptr+1]
    v_add_co_u32 v[vgpr_input_ptr], vcc, s[sgpr_input_ptr], v[vgpr_float4_idx]
    v_addc_co_u32 v[vgpr_input_ptr+1], vcc, 0, v[vgpr_input_ptr+1], vcc
    global_load_dwordx4 v[vgpr_data+0*4+0:vgpr_data+0*4+3], v[vgpr_input_ptr:vgpr_input_ptr+1], off

    v_add_co_u32 v[vgpr_input_ptr], vcc, s[sgpr_stride_bytes], v[vgpr_input_ptr]
    v_addc_co_u32 v[vgpr_input_ptr+1], vcc, 0, v[vgpr_input_ptr+1], vcc
    global_load_dwordx4 v[vgpr_data+1*4+0:vgpr_data+1*4+3], v[vgpr_input_ptr:vgpr_input_ptr+1], off

    s_waitcnt lgkmcnt(0)

    v_mov_b32 v[vgpr_output_ptr+1], s[sgpr_output_ptr+1]
    v_add_co_u32 v[vgpr_output_ptr], vcc, s[sgpr_output_ptr], v[vgpr_float4_idx]
    v_addc_co_u32 v[vgpr_output_ptr+1], vcc, 0, v[vgpr_output_ptr+1], vcc

    v_add_co_u32 v[vgpr_input_ptr], vcc, s[sgpr_stride_bytes], v[vgpr_input_ptr]
    v_addc_co_u32 v[vgpr_input_ptr+1], vcc, 0, v[vgpr_input_ptr+1], vcc

    s_waitcnt vmcnt(1)
    global_store_dwordx4 v[vgpr_output_ptr:vgpr_output_ptr+1], v[vgpr_data+0*4+0:vgpr_data+0*4+3], off
    global_load_dwordx4 v[vgpr_data+0*4+0:vgpr_data+0*4+3], v[vgpr_input_ptr:vgpr_input_ptr+1], off


    v_add_co_u32 v[vgpr_output_ptr], vcc, s[sgpr_stride_bytes], v[vgpr_output_ptr]
    v_addc_co_u32 v[vgpr_output_ptr+1], vcc, 0, v[vgpr_output_ptr+1], vcc

    v_add_co_u32 v[vgpr_input_ptr], vcc, s[sgpr_stride_bytes], v[vgpr_input_ptr]
    v_addc_co_u32 v[vgpr_input_ptr+1], vcc, 0, v[vgpr_input_ptr+1], vcc

    s_waitcnt vmcnt(2)
    global_store_dwordx4 v[vgpr_output_ptr:vgpr_output_ptr+1], v[vgpr_data+1*4+0:vgpr_data+1*4+3], off
    global_load_dwordx4 v[vgpr_data+1*4+0:vgpr_data+1*4+3], v[vgpr_input_ptr:vgpr_input_ptr+1], off


bb0_1:
    s_add_i32 s[sgpr_loop_idx], s[sgpr_loop_idx], 1
    s_cmp_lt_u32 s[sgpr_loop_idx], s[sgpr_num_loops]

    v_add_co_u32 v[vgpr_output_ptr], vcc, s[sgpr_stride_bytes], v[vgpr_output_ptr]
    v_addc_co_u32 v[vgpr_output_ptr+1], vcc, 0, v[vgpr_output_ptr+1], vcc

    v_add_co_u32 v[vgpr_input_ptr], vcc, s[sgpr_stride_bytes], v[vgpr_input_ptr]
    v_addc_co_u32 v[vgpr_input_ptr+1], vcc, 0, v[vgpr_input_ptr+1], vcc

    s_waitcnt vmcnt(2)
    global_store_dwordx4 v[vgpr_output_ptr:vgpr_output_ptr+1], v[vgpr_data+0*4+0:vgpr_data+0*4+3], off
    global_load_dwordx4 v[vgpr_data+0*4+0:vgpr_data+0*4+3], v[vgpr_input_ptr:vgpr_input_ptr+1], off


    v_add_co_u32 v[vgpr_output_ptr], vcc, s[sgpr_stride_bytes], v[vgpr_output_ptr]
    v_addc_co_u32 v[vgpr_output_ptr+1], vcc, 0, v[vgpr_output_ptr+1], vcc

    v_add_co_u32 v[vgpr_input_ptr], vcc, s[sgpr_stride_bytes], v[vgpr_input_ptr]
    v_addc_co_u32 v[vgpr_input_ptr+1], vcc, 0, v[vgpr_input_ptr+1], vcc

    s_waitcnt vmcnt(2)
    global_store_dwordx4 v[vgpr_output_ptr:vgpr_output_ptr+1], v[vgpr_data+1*4+0:vgpr_data+1*4+3], off
    global_load_dwordx4 v[vgpr_data+1*4+0:vgpr_data+1*4+3], v[vgpr_input_ptr:vgpr_input_ptr+1], off


    s_cbranch_scc1 bb0_1


    v_add_co_u32 v[vgpr_output_ptr], vcc, s[sgpr_stride_bytes], v[vgpr_output_ptr]
    v_addc_co_u32 v[vgpr_output_ptr+1], vcc, 0, v[vgpr_output_ptr+1], vcc
    s_waitcnt vmcnt(2)
    global_store_dwordx4 v[vgpr_output_ptr:vgpr_output_ptr+1], v[vgpr_data+0*4+0:vgpr_data+0*4+3], off

    v_add_co_u32 v[vgpr_output_ptr], vcc, s[sgpr_stride_bytes], v[vgpr_output_ptr]
    v_addc_co_u32 v[vgpr_output_ptr+1], vcc, 0, v[vgpr_output_ptr+1], vcc
    s_waitcnt vmcnt(1)
    global_store_dwordx4 v[vgpr_output_ptr:vgpr_output_ptr+1], v[vgpr_data+1*4+0:vgpr_data+1*4+3], off


    s_endpgm
.Lfunc_end0:
    .size   foo_64_2_32_1024, .Lfunc_end0-foo_64_2_32_1024
                                        ; -- End function
    .section    .AMDGPU.csdata
; Kernel info:
; codeLenInByte = 444
; NumSgprs: 16
; NumVgprs: 49
; ScratchSize: 0
; FloatMode: 192
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 1
; VGPRBlocks: 15
; NumSGPRsForWavesPerEU: 16
; NumVGPRsForWavesPerEU: 49
; ReservedVGPRFirst: 0
; ReservedVGPRCount: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 6
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 1
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0

    .ident  "HCC clang version 7.0.0 (ssh://gerritgit/compute/ec/hcc-tot/clang 86791fc4961dc8ffde77bde20d7dfa5e5cbeff5e) (ssh://gerritgit/compute/ec/hcc-tot/llvm c1f9263485c0192d7af512ac2c7dd15d5082538e) (based on HCC 1.2.18272-47899bc-86791fc-c1f9263 )"
    .section    ".note.GNU-stack"
