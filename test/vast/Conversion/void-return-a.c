// RUN: %vast-front -vast-emit-mlir=llvm -vast-snapshot-at="vast-irs-to-llvm" %s
// RUN: %file-check %s -input-file=$(basename %s .c).vast-irs-to-llvm -check-prefix=LLVM

void foo() {
    // LLVM: llvm.call @foo() : () -> ()
    // LLVM: llvm.return
    return foo();
}
