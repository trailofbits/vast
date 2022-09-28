// RUN: vast-cc --ccopts -xc --from-source %s | vast-opt --vast-hl-lower-types --vast-hl-structs-to-tuples --vast-hl-to-ll | FileCheck %s
// REQUIRES: to-ll

// CHECK: llvm.func @fn() {
void fn()
{
    // CHECK: llvm.return
}
// CHECK : }
