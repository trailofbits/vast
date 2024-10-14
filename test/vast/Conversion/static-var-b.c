// RUN: %vast-front -vast-emit-mlir=hl -o - %s | %file-check %s -check-prefix=HL
// RUN: %check-evict-static-locals %s | %file-check %s -check-prefix=EVICTED
// RUN: %check-core-to-llvm %s | %file-check %s -check-prefix=LLVM

// HL: hl.var @zeroinit, <internal>  sc_static
// HL: hl.var @x  sc_static
// HL: hl.ref @x

// EVICTED-DAG: hl.var @zeroinit, <internal> sc_static
// EVICTED-DAG: hl.var @foo.x {context = 0 : i64}, <internal>  sc_static
// EVICTED: ll.func @foo
// EVICTED: hl.ref @foo.x

// LLVM-DAG: llvm.mlir.global internal @zeroinit()
// LLVM-DAG: [[VAR:%[0-9]+]] = llvm.mlir.zero
// LLVM-DAG: llvm.return [[VAR]]
// LLVM-DAG: }
//
// LLVM-DAG: llvm.mlir.global internal @foo.x()
// LLVM-DAG: [[VAR:%[0-9]+]] = llvm.mlir.zero
// LLVM-DAG: llvm.return [[VAR]]
// LLVM-DAG: }
//
// LLVM: llvm.func @foo
// LLVM: llvm.mlir.addressof @foo.x

static int zeroinit;
int foo() {
    static int x;
    return x + zeroinit;
}
