// RUN: vast-cc --ccopts -xc --from-source %s | FileCheck %s
// RUN: vast-cc --ccopts -xc --from-source %s > %t && vast-opt %t | diff -B %t -

// CHECK: hl.var "gstr" : !hl.lvalue<!hl.ptr<!hl.char< const >>>
// CHECK:   hl.const #hl.strlit<"global\n"> : !hl.lvalue<!hl.array<8, !hl.char>>
// CHECK:   ArrayToPointerDecay : !hl.lvalue<!hl.array<8, !hl.char>> -> !hl.lvalue<!hl.ptr<!hl.char>>
const char *gstr = "global\n";
