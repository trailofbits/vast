// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o - | %vast-opt-irs-to-llvm | %file-check %s

// CHECK: llvm.func @fn() -> i32 {
int fn()
{
    // CHECK: [[V:%[0-9]+]] = llvm.mlir.constant(5 : i32) : i32
    // CHECK: llvm.return [[V]] : i32
    return 5;
}
// CHECK : }
