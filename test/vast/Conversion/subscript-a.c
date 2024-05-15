// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o - | %vast-opt-lower-value-categories | %file-check %s

void fn()
{
    // CHECK: [[V0:%[0-9]+]] = ll.alloca : !hl.ptr<memref<3xsi32>>
    // CHECK: [[V4:%[0-9]+]] = hl.initlist {{.*}} : (si32, si32, si32) -> memref<3xsi32>
    // CHECK: ll.store [[V0]], [[V4]] : !hl.ptr<memref<3xsi32>>, memref<3xsi32>
    int arr1[] = { 0, 2, 4 };


    // CHECK: [[V5:%[0-9]+]] = hl.implicit_cast [[V0]] ArrayToPointerDecay : !hl.ptr<memref<3xsi32>> -> !hl.ptr<si32>
    // CHECK: [[V6:%[0-9]+]] = hl.const #core.integer<0> : si32
    // CHECK: [[V7:%[0-9]+]] = ll.subscript [[V5]] at [[[V6]] : si32] : !hl.ptr<si32> -> !hl.ptr<si32>
    // CHECK: [[V8:%[0-9]+]] = ll.load [[V7]] : (!hl.ptr<si32>) -> si32
    (void)arr1[0];
}
