// RUN: %check-emit-abi %s | %file-check %s -check-prefix=EMIT_ABI
// RUN: %check-lower-value-categories %s | %file-check %s -check-prefix=VAL_CAT
// RUN: %check-core-to-llvm %s | %file-check %s -check-prefix=C_LLVM

struct data {
    double x;
    double y;
};

// VAL_CAT:  ll.func @fn{{.*}} ([[A0:%.*]]: !hl.record<@data>) -> none {
// VAL_CAT:    [[V0:%[0-9]+]] = ll.alloca : !hl.ptr<!hl.record<@data>>
// VAL_CAT:    ll.store [[V0]], [[A0]] : !hl.ptr<!hl.record<@data>>, !hl.record<@data>

// EMIT_ABI:  abi.func @vast.abi.fn{{.*}}([[A0:%.*]]: f64, [[A1:%.*]]: f64) -> none {
// EMIT_ABI:    {{.*}} = abi.prologue {
// EMIT_ABI:      [[V3:%[0-9]+]] = abi.direct [[A0]], [[A1]] : f64, f64 -> !hl.record<@data>
// EMIT_ABI:      {{.*}} = abi.yield [[V3]] : !hl.record<@data> -> !hl.record<@data>
// EMIT_ABI:    } : !hl.record<@data>

// C_LLVM:  llvm.func @fn([[A0:%.*]]: f64, [[A1:%.*]]: f64) {
// C_LLVM:    [[V0:%[0-9]+]] = llvm.mlir.undef : !llvm.struct<"data", (f64, f64)>
// C_LLVM:    [[V1:%[0-9]+]] = llvm.insertvalue [[A0]], [[V0]][0] : !llvm.struct<"data", (f64, f64)>
// C_LLVM:    [[V2:%[0-9]+]] = llvm.insertvalue [[A1]], [[V1]][1] : !llvm.struct<"data", (f64, f64)>
// C_LLVM:    [[V3:%[0-9]+]] = llvm.mlir.constant(1 : i64) : i64
// C_LLVM:    [[V4:%[0-9]+]] = llvm.alloca [[V3]] x !llvm.struct<"data", (f64, f64)> : (i64) -> !llvm.ptr
// C_LLVM:    llvm.store [[V2]], [[V4]] : !llvm.struct<"data", (f64, f64)>, !llvm.ptr

void fn(struct data d) {
    d.x + d.y;
}

// VAL_CAT:  [[V5:%[0-9]+]] = ll.load {{.*}} : (!hl.ptr<!hl.record<@data>>) -> !hl.record<@data>
// VAL_CAT:  {{.*}} = hl.call @fn([[V5]]) : (!hl.record<@data>) -> none

// EMIT_ABI:  [[V5:%[0-9]+]] = ll.load %1 : (!hl.ptr<!hl.record<@data>>) -> !hl.record<@data>
// EMIT_ABI:  {{.*}} = abi.call_exec @fn([[V5]]) {
// EMIT_ABI:    [[V7:%[0-9]+]]:2 = abi.call_args {
// EMIT_ABI:      [[V10:%[0-9]+]]:2 = abi.direct [[V5]] : !hl.record<@data> -> f64, f64
// EMIT_ABI:      [[V11:%[0-9]+]]:2 = abi.yield [[V10]]#0, [[V10]]#1 : f64, f64 -> f64, f64
// EMIT_ABI:    } : f64, f64
// EMIT_ABI:    {{.*}} = abi.call @fn([[V7]]#0, [[V7]]#1) : (f64, f64) -> none
// EMIT_ABI:  } : (!hl.record<@data>) -> none


// C_LLVM: {{.*}} = llvm.alloca {{.*}} x !llvm.struct<"data", (f64, f64)> : (i64) -> !llvm.ptr
// C_LLVM: [[V9:%[0-9]+]] = llvm.alloca {{.*}} x !llvm.struct<"data", (f64, f64)> : (i64) -> !llvm.ptr
// C_LLVM: llvm.store {{.*}}, [[V9]] : !llvm.struct<"data", (f64, f64)>, !llvm.ptr
// C_LLVM: [[V10:%[0-9]+]] = llvm.getelementptr [[V9]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"data", (f64, f64)>
// C_LLVM: [[V11:%[0-9]+]] = llvm.load [[V10]] : !llvm.ptr -> f64
// C_LLVM: [[V12:%[0-9]+]] = llvm.getelementptr [[V9]][0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"data", (f64, f64)>
// C_LLVM: [[V13:%[0-9]+]] = llvm.load [[V12]] : !llvm.ptr -> f64
// C_LLVM: llvm.call @fn([[V11]], [[V13]]) : (f64, f64) -> ()
int main()
{
    struct data d = { 0.1, 0.2 };
    fn(d);
}
