// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o - | %vast-opt-lower-value-categories | %file-check %s

void fn()
{
    // CHECK: [[V0:%[0-9]+]] = ll.alloca : !hl.ptr<si32>
    int x = 0;

    // CHECK: [[V2:%[0-9]+]] = ll.alloca : !hl.ptr<si32>
    // CHECK: [[V3:%[0-9]+]] = ll.load [[V0]] : (!hl.ptr<si32>) -> si32
    // CHECK: [[V4:%[0-9]+]] = hl.const #core.integer<1> : si32
    // CHECK: [[V5:%[0-9]+]] = hl.add [[V3]], [[V4]] : (si32, si32) -> si32
    // CHECK: ll.store [[V0]], [[V5]] : !hl.ptr<si32>, si32
    // CHECK: ll.store [[V2]], [[V3]] : !hl.ptr<si32>, si32
    int a = x++;

    // CHECK: [[V6:%[0-9]+]] = ll.alloca : !hl.ptr<si32>
    // CHECK: [[V7:%[0-9]+]] = ll.load [[V0]] : (!hl.ptr<si32>) -> si32
    // CHECK: [[V8:%[0-9]+]] = hl.const #core.integer<1> : si32
    // CHECK: [[V9:%[0-9]+]] = hl.sub [[V7]], [[V8]] : (si32, si32) -> si32
    // CHECK: ll.store [[V0]], [[V9]] : !hl.ptr<si32>, si32
    // CHECK: ll.store [[V6]], [[V9]] : !hl.ptr<si32>, si32
    int b = --x;
}
