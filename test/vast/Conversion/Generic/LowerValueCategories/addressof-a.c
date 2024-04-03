// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o - | %vast-opt-lower-value-categories | %file-check %s

void fn()
{
    // CHECK: [[V0:%[0-9]+]] = ll.alloca : !hl.ptr<si32>
    // CHECK: [[V1:%[0-9]+]] = hl.const #core.integer<0> : si32
    // CHECK: ll.store [[V0]], [[V1]] : !hl.ptr<si32>, si32
    int l = 0;

    // CHECK: [[V2:%[0-9]+]] = ll.alloca : !hl.ptr<!hl.ptr<si32>>
    // CHECK: ll.store [[V2]], [[V0]] : !hl.ptr<!hl.ptr<si32>>, !hl.ptr<si32>
    int *ptr = &l;

    // CHECK: [[V3:%[0-9]+]] = ll.load [[V2]] : (!hl.ptr<!hl.ptr<si32>>) -> !hl.ptr<si32>
    // CHECK: [[V4:%[0-9]+]] = hl.const #core.integer<5> : si32
    // CHECK: ll.store [[V3]], [[V4]] : !hl.ptr<si32>, si32
    *ptr = 5;
}
