// RUN: %check-lower-value-categories %s | %file-check %s -check-prefix=VAL_CAT
// RUN: %check-core-to-llvm %s | %file-check %s -check-prefix=C_LLVM

struct Y;
struct Y { const float x; };

// VAL_CAT: [[V0:%[0-9]+]] = ll.alloca : !hl.ptr<!hl.record<"Y">>
// VAL_CAT: [[V1:%[0-9]+]] = hl.const #core.float<2.000000e+00> : f32
// VAL_CAT: [[V2:%[0-9]+]] = hl.initlist [[V1]] : (f32) -> !hl.record<"Y">
// VAL_CAT: ll.store [[V0]], [[V2]] : !hl.ptr<!hl.record<"Y">>, !hl.record<"Y">

// C_LLVM: [[V1:%[0-9]+]] = llvm.alloca {{.*}} x !llvm.struct<"Y", (f32)> : (i64) -> !llvm.ptr
// C_LLVM: [[V2:%[0-9]+]] = llvm.mlir.constant(2.000000e+00 : f32) : f32
// C_LLVM: [[V3:%[0-9]+]] = llvm.mlir.undef : !llvm.struct<"Y", (f32)>
// C_LLVM: [[V4:%[0-9]+]] = llvm.insertvalue [[V2]], [[V3]][0] : !llvm.struct<"Y", (f32)>
// C_LLVM: llvm.store [[V4]], [[V1]] : !llvm.struct<"Y", (f32)>, !llvm.ptr
int main()
{
    struct Y y = { 2.0f };
    return 0;
}
