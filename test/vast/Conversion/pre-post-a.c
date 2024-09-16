// RUN: %vast-cc1 -vast-emit-mlir-after=vast-lower-value-categories %s -o - | %file-check %s

void fn()
{
    // CHECK: [[V0:%[0-9]+]] = ll.alloca : !hl.ptr<si32>
    int x = 0;

    // CHECK: [[V2:%[0-9]+]] = ll.load [[V0]] : (!hl.ptr<si32>) -> si32
    // CHECK: [[V3:%[0-9]+]] = hl.const #core.integer<1> : si32
    // CHECK: [[V4:%[0-9]+]] = hl.add [[V2]], [[V3]] : (si32, si32) -> si32
    // CHECK: ll.store [[V0]], [[V4]] : !hl.ptr<si32>, si32
    ++x;

    // CHECK: [[V5:%[0-9]+]] = ll.load [[V0]] : (!hl.ptr<si32>) -> si32
    // CHECK: [[V6:%[0-9]+]] = hl.const #core.integer<1> : si32
    // CHECK: [[V7:%[0-9]+]] = hl.add [[V5]], [[V6]] : (si32, si32) -> si32
    // CHECK: ll.store [[V0]], [[V7]] : !hl.ptr<si32>, si32
    x++;

    // CHECK: [[V8:%[0-9]+]] = ll.load [[V0]] : (!hl.ptr<si32>) -> si32
    // CHECK: [[V9:%[0-9]+]] = hl.const #core.integer<1> : si32
    // CHECK: [[V10:%[0-9]+]] = hl.sub [[V8]], [[V9]] : (si32, si32) -> si32
    // CHECK: ll.store [[V0]], [[V10]] : !hl.ptr<si32>, si32
    --x;

    // CHECK: [[V11:%[0-9]+]] = ll.load [[V0]] : (!hl.ptr<si32>) -> si32
    // CHECK: [[V12:%[0-9]+]] = hl.const #core.integer<1> : si32
    // CHECK: [[V13:%[0-9]+]] = hl.sub [[V11]], [[V12]] : (si32, si32) -> si32
    // CHECK: ll.store [[V0]], [[V13]] : !hl.ptr<si32>, si32
    x--;
}
