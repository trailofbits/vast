// RUN: vast-cc --ccopts -xc --from-source %s | vast-opt --vast-hl-lower-types --vast-hl-structs-to-tuples --vast-hl-to-ll | FileCheck %s
// REQUIRES: to-ll

// CHECK: llvm.func @fn() -> i32 {
int fn()
{
    // CHECK: [[V:%[0-9]+]] = llvm.mlir.constant(5 : ui4) : i32
    // CHECK: llvm.return [[V]] : i32
    return 5;
}
// CHECK : }
