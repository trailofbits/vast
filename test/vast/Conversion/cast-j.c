// RUN: %vast-cc1 -triple x86_64-unknown-linux-gnu -vast-emit-mlir=hl %s -o %t.mlir
// RUN: %file-check --input-file=%t.mlir %s -check-prefix=HL
// RUN: %vast-cc1 -triple x86_64-unknown-linux-gnu -vast-emit-mlir=llvm %s -o %t.mlir
// RUN: %file-check --input-file=%t.mlir %s -check-prefix=MLIR
// RUN: %vast-cc1 -triple x86_64-unknown-linux-gnu -vast-emit-llvm %s -o %t.ll
// RUN: %file-check --input-file=%t.ll %s -check-prefix=LLVM

void null_to_ptr() {
    void *p = 0;
}

// HL: hl.func @null_to_ptr
// HL:    hl.implicit_cast {{.*}} NullToPointer : !hl.int -> !hl.ptr<!hl.void>
// HL: }

// MLIR: llvm.func @null_to_ptr
// MLIR:    llvm.mlir.zero
// MLIR: }

// LLVM: define void @null_to_ptr
// LLVM:    store ptr null, ptr
// LLVM: }
