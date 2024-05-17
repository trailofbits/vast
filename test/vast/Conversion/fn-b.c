// RUN: %check-core-to-llvm %s | %file-check %s -check-prefix=C_LLVM

// C_LLVM: llvm.func @fn() -> i32 {
// C_LLVM:   [[V:%[0-9]+]] = llvm.mlir.constant(5 : i32) : i32
// C_LLVM:   llvm.return [[V]] : i32
// C_LLVM: }
int fn()
{
    return 5;
}
