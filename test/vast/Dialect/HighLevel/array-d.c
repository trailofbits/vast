// RUN: vast-cc --ccopts -xc --from-source %s | FileCheck %s
// RUN: vast-cc --ccopts -xc --from-source %s > %t && vast-opt %t | diff -B %t -

int main() {
    // CHECK: hl.var @arr : !hl.const.array<3, !hl.int>
    int arr[3];
    // CHECK: hl.var @v0 : !hl.int = {
    // CHECK:   [[V1:%[0-9]+]] = hl.declref @arr : !hl.const.array<3, !hl.int>
    // CHECK:   [[V2:%[0-9]+]] = hl.implicit_cast [[V1]] ArrayToPointerDecay : !hl.const.array<3, !hl.int> -> !hl.ptr<!hl.int>
    // CHECK:   [[V3:%[0-9]+]] = hl.constant.int 0 : !hl.int
    // CHECK:   hl.subscript [[V2]] at {{.*}}[[V3]] : !hl.int] : !hl.ptr<!hl.int> -> !hl.int
    // CHECK: }
    int v0 = arr[0];

    // CHECK: hl.var @v2 : !hl.int = {
    // CHECK:   [[V1:%[0-9]+]] = hl.declref @arr : !hl.const.array<3, !hl.int>
    // CHECK:   [[V2:%[0-9]+]] = hl.implicit_cast [[V1]] ArrayToPointerDecay : !hl.const.array<3, !hl.int> -> !hl.ptr<!hl.int>
    // CHECK:   [[V3:%[0-9]+]] = hl.declref @idx : !hl.int
    // CHECK:   [[V4:%[0-9]+]] = hl.implicit_cast [[V3]] LValueToRValue : !hl.int -> !hl.int
    // CHECK:   hl.subscript [[V2]] at {{.*}}[[V4]] : !hl.int] : !hl.ptr<!hl.int> -> !hl.int
    // CHECK: }
    int idx = 2;
    int v2 = arr[idx];
}
