// RUN: %check-fn-args-to-alloca %s | %file-check %s -check-prefix=ARGS_ALLOCA
// RUN: %check-lower-value-categories %s | %file-check %s -check-prefix=VAL_CAT
// RUN: %check-core-to-llvm %s | %file-check %s -check-prefix=C_LLVM

// ARGS_ALLOCA:  ll.func @fn external ([[A1:%.*]]: si32, [[A2:%.*]]: si32) -> si32 {
// ARGS_ALLOCA:    {{.*}} = ll.arg_alloca [[A1]] : (si32) -> !hl.lvalue<si32>
// ARGS_ALLOCA:    {{.*}} = ll.arg_alloca [[A2]] : (si32) -> !hl.lvalue<si32>

// VAL_CAT:  ll.func @fn external ([[A0:%.*]]: si32, [[A1:%.*]]: si32) -> si32 {
// VAL_CAT:    [[V0:%[0-9]+]] = ll.alloca : !hl.ptr<si32>
// VAL_CAT:    ll.store [[V0]], [[A0]] : !hl.ptr<si32>, si32
// VAL_CAT:    [[V1:%[0-9]+]] = ll.alloca : !hl.ptr<si32>
// VAL_CAT:    ll.store [[V1]], [[A1]] : !hl.ptr<si32>, si32
// VAL_CAT:    {{.*}} = ll.load [[V0]] : (!hl.ptr<si32>) -> si32
// VAL_CAT:    {{.*}} = ll.load [[V1]] : (!hl.ptr<si32>) -> si32

// C_LLVM:  llvm.func @fn([[A0:%.*]]: i32, [[A1:%.*]]: i32) -> i32 {
// C_LLVM:    [[V1:%[0-9]+]] = llvm.alloca {{.*}} x i32 : (i64) -> !llvm.ptr
// C_LLVM:    llvm.store [[A0]], [[V1]] : i32, !llvm.ptr
// C_LLVM:    [[V3:%[0-9]+]] = llvm.alloca {{.*}} x i32 : (i64) -> !llvm.ptr
// C_LLVM:    llvm.store [[A1]], [[V3]] : i32, !llvm.ptr

int fn(int arg0, int arg1)
{
    // CHECK: %0 = ll.arg_alloca %arg0 : (!hl.int) -> !hl.lvalue<!hl.int>
    // CHECK: %1 = ll.arg_alloca %arg1 : (!hl.int) -> !hl.lvalue<!hl.int>
    // CHECK: {{.*}} = hl.ref %0 : (!hl.lvalue<!hl.int>) -> !hl.lvalue<!hl.int>
    // CHECK: {{.*}} = hl.ref %1 : (!hl.lvalue<!hl.int>) -> !hl.lvalue<!hl.int>
    return arg0 + arg1;
}
