// RUN: %check-lower-value-categories %s | %file-check %s -check-prefix=VAL_CAT
// RUN: %check-core-to-llvm %s | %file-check %s -check-prefix=C_LLVM

struct X { int a; struct X *x; };

// VAL_CAT: [[V0:%[0-9]+]] = ll.alloca : !hl.ptr<!hl.record<"X">>
// VAL_CAT: [[V3:%[0-9]+]] = hl.implicit_cast {{.*}} NullToPointer : si32 -> !hl.ptr<!hl.record<"X">>
// VAL_CAT: [[V4:%[0-9]+]] = hl.initlist {{.*}}, [[V3]] : (si32, !hl.ptr<!hl.record<"X">>) -> !hl.record<"X">
// VAL_CAT: ll.store [[V0]], [[V4]] : !hl.ptr<!hl.record<"X">>, !hl.record<"X">

// C_LLVM: [[V1:%[0-9]+]] = llvm.alloca {{.*}} x !llvm.struct<"X", (i32, ptr)> : (i64) -> !llvm.ptr
// C_LLVM: [[V2:%[0-9]+]] = llvm.mlir.constant(2 : i32) : i32
// C_LLVM: [[V3:%[0-9]+]] = llvm.mlir.constant(0 : i32) : i32
// C_LLVM: [[V4:%[0-9]+]] = llvm.mlir.zero : !llvm.ptr
// C_LLVM: [[V5:%[0-9]+]] = llvm.mlir.undef : !llvm.struct<"X", (i32, ptr)>
// C_LLVM: [[V6:%[0-9]+]] = llvm.insertvalue [[V2]], [[V5]][0] : !llvm.struct<"X", (i32, ptr)>
// C_LLVM: [[V7:%[0-9]+]] = llvm.insertvalue [[V4]], [[V6]][1] : !llvm.struct<"X", (i32, ptr)>
// C_LLVM: llvm.store [[V7]], [[V1]] : !llvm.struct<"X", (i32, ptr)>, !llvm.ptr
int main()
{
    struct X x = { 2, 0 };
    return 0;
}
