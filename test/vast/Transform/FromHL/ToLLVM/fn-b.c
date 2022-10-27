// RUN: vast-cc --ccopts -xc --from-source %s | vast-opt --vast-hl-lower-types --vast-hl-to-ll-vars --vast-core-to-llvm | FileCheck %s

// REQUIRES: funcop-lowering

// CHECK: llvm.func @fn() -> i32 {
int fn()
{
    // CHECK: [[V:%[0-9]+]] = llvm.mlir.constant(5 : i32) : i32
    // CHECK: llvm.return [[V]] : i32
    return 5;
}
// CHECK : }
