// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o - | %vast-opt-lower-value-categories | %file-check %s

void fn(int x, int v)
{
    // CHECK: [[V0:%[0-9]+]] = ll.alloca : !hl.ptr<si32>
    // CHECK: ll.store [[V0]], %arg0 : !hl.ptr<si32>, si32
    // CHECK: [[V1:%[0-9]+]] = ll.alloca : !hl.ptr<si32>
    // CHECK: ll.store [[V1]], %arg1 : !hl.ptr<si32>, si32

    // CHECK: [[V2:%[0-9]+]] = ll.load [[V1]] : (!hl.ptr<si32>) -> si32
    // CHECK: [[V3:%[0-9]+]] = ll.load [[V0]] : (!hl.ptr<si32>) -> si32
    // CHECK: [[V4:%[0-9]+]] = hl.add [[V3]], [[V2]] : (si32, si32) -> si32
    // CHECK: ll.store [[V0]], [[V4]] : !hl.ptr<si32>, si32
    x += v;

    // CHECK: [[V5:%[0-9]+]] = ll.load [[V1]] : (!hl.ptr<si32>) -> si32
    // CHECK: [[V6:%[0-9]+]] = ll.load [[V0]] : (!hl.ptr<si32>) -> si32
    // CHECK: [[V7:%[0-9]+]] = hl.mul [[V6]], [[V5]] : (si32, si32) -> si32
    // CHECK: ll.store [[V0]], [[V7]] : !hl.ptr<si32>, si32
    x *= v;
}
