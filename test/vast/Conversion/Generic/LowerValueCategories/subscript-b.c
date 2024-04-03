// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o - | %vast-opt-lower-value-categories | %file-check %s

// CHECK:  [[G:%[0-9]+]] = hl.var "arr1" : !hl.ptr<memref<3xsi32>> = {
// CHECK:    [[GV4:%[0-9]+]] = hl.initlist {{.*}} : (si32, si32, si32) -> memref<3xsi32>
// CHECK:    hl.value.yield [[GV4]] : memref<3xsi32>
int arr1[] = { 0, 2, 4 };

void fn()
{
    // CHECK: [[V1:%[0-9]+]] = hl.globref "arr1" : !hl.ptr<memref<3xsi32>>
    // CHECK: [[V2:%[0-9]+]] = hl.implicit_cast [[V1]] ArrayToPointerDecay : !hl.ptr<memref<3xsi32>> -> !hl.ptr<si32>
    // CHECK: [[V3:%[0-9]+]] = hl.const #core.integer<0> : si32
    // CHECK: [[V4:%[0-9]+]] = ll.subscript [[V2]] at [[[V3]] : si32] : !hl.ptr<si32> -> !hl.ptr<si32>
    // CHECK: [[V5:%[0-9]+]] = ll.load [[V4]] : (!hl.ptr<si32>) -> si32
    (void)arr1[0];
}
