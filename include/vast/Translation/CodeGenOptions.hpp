// Copyright (c) 2021-present, Trail of Bits, Inc.

#pragma once

namespace vast::cg {

    struct codegen_options {
        bool verbose_diagnostics = true;
        // forwarded options form clang codegen
        unsigned int coverage_mapping = false;
        unsigned int keep_static_consts = false;
        unsigned int patchable_function_entry_count  = 0;
        unsigned int patchable_function_entry_offset = 0;
        unsigned int no_use_jump_tables = false;
        unsigned int no_inline_line_tables = false;
        unsigned int packed_stack = false;
        unsigned int warn_stack_size = false;
        unsigned int strict_return = false;
        unsigned int optimization_level = 0;

        bool should_emit_lifetime_markers = false;
    };

}  // namespace vast::cg
