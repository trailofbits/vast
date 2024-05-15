// RUN: %vast-front -vast-emit-mlir-after=vast-irs-to-llvm %s -o %t.mlir
// RUN: %file-check --input-file=%t.mlir %s -check-prefix=LLVM

void foo() {
    // LLVM: llvm.call @foo() : () -> ()
    // LLVM: llvm.return
    return foo();
}
