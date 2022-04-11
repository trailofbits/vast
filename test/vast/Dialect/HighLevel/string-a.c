// RUN: vast-cc --ccopts -xc --from-source %s | FileCheck %s
// RUN: vast-cc --ccopts -xc --from-source %s > %t && vast-opt %t | diff -B %t -

// CHECK: hl.var "gstr" : !hl.lvalue<!hl.ptr<!hl.char<const>>>
// CHECK: hl.constant.string "global" : !hl.const.array<7, !hl.char>
// CHECK: ArrayToPointerDecay : !hl.const.array<7, !hl.char> -> !hl.ptr<!hl.char>
const char *gstr = "global";
