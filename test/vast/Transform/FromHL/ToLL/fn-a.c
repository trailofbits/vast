// RUN: vast-cc --ccopts -xc --from-source %s | vast-opt --vast-hl-lower-types --vast-hl-structs-to-tuples --vast-hl-to-ll | FileCheck %s

// CHECK: func @fn() -> none {
void fn()
{
    // CHECK: llvm.return
}
// CHECK : }
