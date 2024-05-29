// RUN: %check-lower-value-categories %s | %file-check %s -check-prefix=VAL_CAT
// RUN: %check-core-to-llvm %s | %file-check %s -check-prefix=C_LLVM

void fn()
{
    // VAL_CAT: [[V0:%[0-9]+]] = ll.alloca : !hl.ptr<si32>
    // VAL_CAT: [[V1:%[0-9]+]] = hl.const #core.integer<0> : si32
    // VAL_CAT: ll.store [[V0]], [[V1]] : !hl.ptr<si32>, si32

    // VAL_CAT: [[V2:%[0-9]+]] = ll.alloca : !hl.ptr<!hl.ptr<si32>>
    // VAL_CAT: ll.store [[V2]], [[V0]] : !hl.ptr<!hl.ptr<si32>>, !hl.ptr<si32>

    // VAL_CAT: [[V3:%[0-9]+]] = ll.load [[V2]] : (!hl.ptr<!hl.ptr<si32>>) -> !hl.ptr<si32>
    // VAL_CAT: [[V4:%[0-9]+]] = hl.const #core.integer<5> : si32
    // VAL_CAT: ll.store [[V3]], [[V4]] : !hl.ptr<si32>, si32

    // C_LLVM: [[V1:%[0-9]+]] = llvm.alloca {{.*}} x i32 : (i64) -> !llvm.ptr
    // C_LLVM: [[V2:%[0-9]+]] = llvm.mlir.constant(0 : i32) : i32
    // C_LLVM: llvm.store [[V2]], [[V1]] : i32, !llvm.ptr

    // C_LLVM: [[V4:%[0-9]+]] = llvm.alloca {{.*}} x !llvm.ptr : (i64) -> !llvm.ptr
    // C_LLVM: llvm.store [[V1]], [[V4]] : !llvm.ptr, !llvm.ptr

    // C_LLVM: [[V5:%[0-9]+]] = llvm.load [[V4]] : !llvm.ptr -> !llvm.ptr
    // C_LLVM: [[V6:%[0-9]+]] = llvm.mlir.constant(5 : i32) : i32
    // C_LLVM: llvm.store [[V6]], [[V5]] : i32, !llvm.ptr

    int l = 0;

    int *ptr = &l;

    *ptr = 5;
}
