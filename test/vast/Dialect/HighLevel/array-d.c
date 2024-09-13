// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o - | %file-check %s
// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o %t && %vast-opt %t | diff -B %t -

// TODO decl.ref registers

int main() {
    // CHECK: hl.var @arr : !hl.lvalue<!hl.array<3, !hl.int>>
    int arr[3];
    // CHECK: hl.var @v0 : !hl.lvalue<!hl.int> = {
    // CHECK:   [[V1:%[0-9]+]] = hl.ref @arr
    // CHECK:   [[V2:%[0-9]+]] = hl.implicit_cast [[V1]] ArrayToPointerDecay : !hl.lvalue<!hl.array<3, !hl.int>> -> !hl.ptr<!hl.int>
    // CHECK:   [[V3:%[0-9]+]] = hl.const #core.integer<0> : !hl.int
    // CHECK:   hl.subscript [[V2]] at {{.*}}[[V3]] : !hl.int] : !hl.ptr<!hl.int> -> !hl.lvalue<!hl.int>
    // CHECK: }
    int v0 = arr[0];

    // CHECK: hl.var @idx : !hl.lvalue<!hl.int>
    int idx = 2;
    // CHECK: hl.var @v2 : !hl.lvalue<!hl.int> = {
    // CHECK:   [[V1:%[0-9]+]] = hl.ref @arr
    // CHECK:   [[V2:%[0-9]+]] = hl.implicit_cast [[V1]] ArrayToPointerDecay : !hl.lvalue<!hl.array<3, !hl.int>> -> !hl.ptr<!hl.int>
    // CHECK:   [[V3:%[0-9]+]] = hl.ref @idx
    // CHECK:   [[V4:%[0-9]+]] = hl.implicit_cast [[V3]] LValueToRValue : !hl.lvalue<!hl.int> -> !hl.int
    // CHECK:   hl.subscript [[V2]] at {{.*}}[[V4]] : !hl.int] : !hl.ptr<!hl.int> -> !hl.lvalue<!hl.int>
    // CHECK: }
    int v2 = arr[idx];
}
