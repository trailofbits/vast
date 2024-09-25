// RUN: %vast-front -vast-emit-mlir=hl -o - %s | %file-check %s -check-prefix=HL
// RUN: %check-evict-static-locals %s | %file-check %s -check-prefix=EVICTED
// RUN: %check-core-to-llvm %s | %file-check %s -check-prefix=LLVM

// HL: hl.var @x  sc_static
// HL: hl.ref @x

// EVICTED: hl.var @foo.x {context = 0 : i64}, <internal>  sc_static
// EVICTED: ll.func @foo
// EVICTED: hl.ref @foo.x

// LLVM: llvm.mlir.global internal @foo.x()
// LLVM: llvm.func @foo
// LLVM: llvm.mlir.addressof @foo.x
int foo() {
    static int x = 5;
    return x++;
}
