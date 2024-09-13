// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o - | %file-check %s
// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o %t && %vast-opt %t | diff -B %t -

// CHECK-LABEL: hl.func @_Z11cast_cstylev
void cast_cstyle()
{
    int i = 0;
    // CHECK: hl.var @i : !hl.lvalue<!hl.int>
    short s = (short)i;
    // CHECK: hl.var @s : !hl.lvalue<!hl.short>
    // CHECK: [[V1:%[0-9]+]] = hl.ref @i
    // CHECK: [[V2:%[0-9]+]] = hl.implicit_cast [[V1]] LValueToRValue : !hl.lvalue<!hl.int> -> !hl.int
    // CHECK: [[V3:%[0-9]+]] = hl.implicit_cast [[V2]] IntegralCast : !hl.int -> !hl.short
    // CHECK: [[V4:%[0-9]+]] = hl.cstyle_cast [[V3]] NoOp : !hl.short -> !hl.short
    // CHECK: hl.value.yield [[V4]] : !hl.short
}
