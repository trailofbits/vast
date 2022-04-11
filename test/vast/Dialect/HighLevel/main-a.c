// RUN: vast-cc --ccopts -xc --from-source %s | FileCheck %s
// RUN: vast-cc --ccopts -xc --from-source %s > %t && vast-opt %t | diff -B %t -

// CHECK: hl.typedef "__int128_t" : !hl.int128
// CHECK: hl.typedef "__uint128_t" : !hl.int128<unsigned>

// CHECK: hl.record "struct __NSConstantString_tag"
// CHECK:   hl.field "isa" : !hl.ptr<!hl.int<const>>
// CHECK:   hl.field "flags" : !hl.int
// CHECK:   hl.field "str" : !hl.ptr<!hl.char<const>>
// CHECK:   hl.field "length" : !hl.long

// CHECK: hl.typedef "__NSConstantString" : !hl.named_type<"struct __NSConstantString_tag">
// CHECK: hl.typedef "__builtin_ms_va_list" : !hl.ptr<!hl.char>

// CHECK: hl.record "struct __va_list_tag"
// CHECK:    hl.field "gp_offset" : !hl.int<unsigned>
// CHECK:    hl.field "fp_offset" : !hl.int<unsigned>
// CHECK:    hl.field "overflow_arg_area" : !hl.ptr<!hl.void>
// CHECK:    hl.field "reg_save_area" : !hl.ptr<!hl.void>

// CHECK: hl.typedef "__builtin_va_list" : !hl.const.array<1, !hl.named_type<"struct __va_list_tag">>

// CHECK-LABEL: func @main() -> !hl.int
int main() {}
