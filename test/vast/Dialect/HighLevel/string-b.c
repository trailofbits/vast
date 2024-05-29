// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o - | %file-check %s
// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o %t && %vast-opt %t | diff -B %t -

// CHECK: hl.var @gstr : !hl.lvalue<!hl.ptr<!hl.char< const >>>
// CHECK:   hl.const "global\0A" : !hl.lvalue<!hl.array<8, !hl.char>>
// CHECK:   ArrayToPointerDecay : !hl.lvalue<!hl.array<8, !hl.char>> -> !hl.ptr<!hl.char>
const char *gstr = "global\n";
