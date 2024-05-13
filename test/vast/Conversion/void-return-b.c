// RUN: %vast-front -vast-emit-mlir=llvm -vast-snapshot-at="vast-irs-to-llvm" %s
// RUN: %file-check %s -input-file=$(basename %s .c).vast-irs-to-llvm -check-prefix=LLVM

// REQUIRES: hl.stmt.expr

void foo() {
    return ( { foo(); } );
}
